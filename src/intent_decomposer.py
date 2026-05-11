"""
intent_decomposer.py

Breaks a high-level user intent into concrete BraneScript sub-tasks,
each of which can be independently retrieved from the knowledge base.

Drop-in for your existing pipeline: call decompose_intent() before the
retriever, then pass the expanded queries to multi_query_retrieve().
"""

from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# 1. Decomposition prompt
#    Instructs the LLM to split the intent into BraneScript primitives.
#    Output is a numbered list — easy to parse, no JSON needed.
# ---------------------------------------------------------------------------
DECOMPOSE_TEMPLATE = """
You are a BraneScript expert. Your job is to break a user's high-level intent
into a short list of concrete, retrievable BraneScript sub-tasks.

Each sub-task should correspond to ONE of these primitive operations:
- Defining a function (func keyword, parameters, return type)
- Importing / calling a package
- Reading or referencing a dataset
- Defining a workflow (top-level orchestration)
- Variable assignment or type usage (let, :=, unit)
- Control flow (if/else, loops)

Rules:
- Output ONLY a numbered list, one sub-task per line.
- Each line must be a short search query (5-15 words) suitable for
  retrieving BraneScript documentation.
- Do NOT output code. Do NOT explain. Max 6 sub-tasks.
- Phrase each sub-task as if you were searching the BraneScript manual.

USER INTENT:
{intent}

SUB-TASKS:
"""

# ---------------------------------------------------------------------------
# 2. Rewriter prompt
#    Turns each sub-task into a clean doc-search query.
#    This handles vocabulary mismatch (user says "analyze", docs say "process").
# ---------------------------------------------------------------------------
REWRITE_TEMPLATE = """
You are helping search BraneScript documentation. Rewrite the following
sub-task as a short, precise documentation search query. Use BraneScript
terminology where possible (e.g. func, let, workflow, package, unit, import).

SUB-TASK: {subtask}

SEARCH QUERY (one line only):
"""


class IntentDecomposer:
    """
    Decomposes a user intent into sub-tasks, rewrites each as a search query, then retrieves docs for all of them in one pass.
    Usage:
        decomposer = IntentDecomposer(llm, db)
        context, subtasks = decomposer.run("I want to analyze heart-disease
                                            data using package A")
    """

    def __init__(self, llm: Ollama, db: Chroma, k_per_subtask: int = 3):
        self.llm = llm
        self.db = db
        self.k = k_per_subtask

        # Chain: intent -> numbered list of sub-tasks
        self.decompose_chain = (
            PromptTemplate.from_template(DECOMPOSE_TEMPLATE)
            | llm
            | StrOutputParser()
        )

        # Chain: sub-task -> cleaned search query
        self.rewrite_chain = (
            PromptTemplate.from_template(REWRITE_TEMPLATE)
            | llm
            | StrOutputParser()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_subtasks(self, raw: str) -> list[str]:
        """Parse '1. foo\n2. bar' into ['foo', 'bar']."""
        subtasks = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip leading "1." / "1)" / "-" / "*"
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.",
                           "1)", "2)", "3)", "4)", "5)", "6)",
                           "-", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)
        return subtasks[:6]  # hard cap

    def _retrieve_for_query(self, query: str) -> list:
        """Vector similarity search for a single query."""
        return self.db.similarity_search(query, k=self.k)

    def _deduplicate(self, docs: list) -> list:
        """Remove duplicate chunks by page_content."""
        seen = set()
        unique = []
        for doc in docs:
            key = doc.page_content[:120]  # fingerprint on first 120 chars
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, intent: str) -> list[str]:
        """Return a list of sub-task strings from the user's intent."""
        raw = self.decompose_chain.invoke({"intent": intent})
        subtasks = self._parse_subtasks(raw)
        print(f"\n📋 Decomposed into {len(subtasks)} sub-tasks:")
        for i, s in enumerate(subtasks, 1):
            print(f"   {i}. {s}")
        return subtasks

    def rewrite_queries(self, subtasks: list[str]) -> list[str]:
        """Rewrite sub-tasks into doc-search queries to fix vocab mismatch."""
        queries = []
        for st in subtasks:
            q = self.rewrite_chain.invoke({"subtask": st}).strip()
            # Fallback: use original sub-task if rewrite is empty/garbage
            queries.append(q if len(q) > 5 else st)
        return queries

    def retrieve_all(self, queries: list[str]) -> list:
        """Retrieve and deduplicate docs for all queries."""
        all_docs = []
        for q in queries:
            docs = self._retrieve_for_query(q)
            all_docs.extend(docs)
        return self._deduplicate(all_docs)

    def run(self, intent: str) -> tuple[str, list[str]]:
        """
        Full pipeline: intent -> subtasks -> queries -> deduplicated docs.

        Returns:
            context  (str)       : formatted context string for the prompt
            subtasks (list[str]) : the decomposed sub-tasks (for logging)
        """
        subtasks = self.decompose(intent)
        queries = self.rewrite_queries(subtasks)

        print(f"\n🔍 Retrieval queries:")
        for q in queries:
            print(f"   • {q}")

        docs = self.retrieve_all(queries)
        print(f"\n📄 Retrieved {len(docs)} unique chunks\n")

        context = "\n\n---\n\n".join(doc.page_content for doc in docs)
        return context, subtasks
