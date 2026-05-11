"""
Responsibility: HIGH LEVEL TASK BREAKDOWN + LANGUAGE SPEC RETRIEVAL

Two jobs:
  1. Break the user's intent into concrete BraneScript sub-tasks (plain English).
  2. Retrieve only the relevant language spec chunks for those sub-tasks,
     capped at a token budget to avoid overflowing llama3 context window.
"""

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma

MAX_LANG_CONTEXT_CHARS = 1500 * 4    # ~6000 chars ≈ 1500 tokens

# ---------------------------------------------------------------------------
# Prompt 1: Decomposition: breaks user intent into sub-tasks in plain English, no code is generated at this stage
# ---------------------------------------------------------------------------
DECOMPOSE_TEMPLATE = """You are a BraneScript expert. Break the user's intent
into a short numbered list of concrete BraneScript sub-tasks.

Each sub-task maps to ONE primitive:
- Defining a function (func keyword, parameters, return type)
- Importing / calling a package
- Reading or referencing a dataset
- Defining a workflow (top-level orchestration)
- Variable assignment or type usage (let, :=, unit)
- Control flow (if/else, loops)

Rules:
- Output ONLY a numbered list, one sub-task per line.
- Each line: 5-15 words, phrased as a BraneScript manual search query.
- No code. No explanations. Max 10 sub-tasks.

USER INTENT:
{intent}

SUB-TASKS:"""

# ---------------------------------------------------------------------------
# Prompt 2: Query rewriter: translates English sub-tasks into BraneScript-vocabulary search terms.
# ---------------------------------------------------------------------------
REWRITE_TEMPLATE = """Rewrite the following sub-task as a short BraneScript
documentation search query (one line, 5-10 words).
Use BraneScript terms where possible: func, let, workflow, package, unit, import.

SUB-TASK: {subtask}

SEARCH QUERY:"""


class IntentDecomposer:
    """
    High level task breakdown + language spec retrieval.

    Usage:
        decomposer = IntentDecomposer(llm, lang_db)
        lang_context, subtasks = decomposer.run("I want to analyze heart-disease data")
    """

    def __init__(self, llm: Ollama, lang_db: Chroma, k_per_subtask: int = 3):
        self.llm = llm
        self.lang_db = lang_db
        self.k = k_per_subtask

        self.decompose_chain = (
            PromptTemplate.from_template(DECOMPOSE_TEMPLATE)
            | llm
            | StrOutputParser()
        )
        self.rewrite_chain = (
            PromptTemplate.from_template(REWRITE_TEMPLATE)
            | llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_subtasks(self, raw: str) -> list[str]:
        subtasks = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.",
                           "1)", "2)", "3)", "4)", "5)", "6)",
                           "-", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)
            
            subtasks = [s for s in subtasks if len(s) < 80 and not s.endswith(":")]
        return subtasks[:6]

    def _rewrite(self, subtask: str) -> str:
        """Rewrite one sub-task into a BraneScript-vocabulary search query."""
        q = self.rewrite_chain.invoke({"subtask": subtask}).strip()
        return q if len(q) > 5 else subtask    # fallback to original if garbage

    def _retrieve(self, query: str) -> list:
        return self.lang_db.similarity_search(query, k=self.k)

    def _deduplicate(self, docs: list) -> list:
        seen, unique = set(), []
        for doc in docs:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _apply_token_budget(self, docs: list) -> list:
        """Keep chunks until we hit the character budget."""
        total, kept = 0, []
        for doc in docs:
            size = len(doc.page_content)
            if total + size > MAX_LANG_CONTEXT_CHARS:
                break
            kept.append(doc)
            total += size
        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, intent: str) -> list[str]:
        """Break the intent into a list of plain-English sub-tasks."""
        raw = self.decompose_chain.invoke({"intent": intent})
        subtasks = self._parse_subtasks(raw)
        print(f"\n📋 Task breakdown ({len(subtasks)} sub-tasks):")
        for i, s in enumerate(subtasks, 1):
            print(f"   {i}. {s}")
        return subtasks

    def retrieve_lang_context(self, subtasks: list[str]) -> str:
        """Rewrite sub-tasks into search queries, then it retrieves lang spec chunks."""
        print("\n🔍 Language spec retrieval:")
        all_docs = []
        for st in subtasks:
            query = self._rewrite(st)
            print(f"   • {query}")
            all_docs.extend(self._retrieve(query))

        docs = self._apply_token_budget(self._deduplicate(all_docs))
        print(f"   → {len(docs)} unique chunks within token budget")
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def run(self, intent: str) -> tuple[str, list[str]]:
        """
        Full flow: intent → sub-tasks → lang spec context.

        Returns:
            lang_context  (str)       : syntax docs for the prompt
            subtasks      (list[str]) : sub-tasks for prompt structure + pkg retrieval
        """
        subtasks = self.decompose(intent)
        lang_context = self.retrieve_lang_context(subtasks)
        return lang_context, subtasks
