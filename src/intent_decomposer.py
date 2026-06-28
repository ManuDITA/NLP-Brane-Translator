"""
Responsibility: HIGH LEVEL TASK BREAKDOWN

Breaks the user's intent into concrete BraneScript sub-tasks (plain English).
Language spec context is no longer retrieved here — syntax_reference.md is
always injected directly into the prompt by pipeline.py.
"""

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from utils import strip_thinking_tokens

# ---------------------------------------------------------------------------
# Prompt 1: Decomposition: breaks user intent into sub-tasks in plain English, no code is generated at this stage
# ---------------------------------------------------------------------------
DECOMPOSE_TEMPLATE = """You are a BraneScript expert. Break the user's intent
into a short numbered list of concrete BraneScript sub-tasks.

Each sub-task maps to ONE primitive operation:
- Defining a function (func keyword, parameters, return type)
- Importing a package (import keyword)
- Calling a package function or external function
- Defining a workflow (top-level statements)
- Variable assignment and declaration (let, :=)
- If/else conditional branches
- For-loops and while-loops
- Parallel execution blocks with merge strategies
- Return statements (returning values or early exit)
- Arrays (creation, indexing, operations)
- Classes (class definition, properties, methods)
- Object instantiation (new keyword)
- Routing execution to a named node, site, or location using #[on("name")] attribute
  (trigger this whenever the user says: "on node X", "on site X", "at location X", "run on X", "execute on X")
- Projections and property access (dot notation, e.g., obj.property)
- Expression statements (calling functions for side-effects)
- Block statements for scoping
- Break and continue for loop control

Rules:
- Output ONLY a numbered list, one sub-task per line.
- Each line: 5-15 words, phrased as a BraneScript manual search query.
- No code. No explanations. Max 12 sub-tasks.
- Focus ONLY on BraneScript constructs needed to pass data to the package function.
  Do NOT generate subtasks for logic that the package function handles internally
  (e.g. do not say "compute risk score", "validate input", "handle edge cases" —
  those are inside the package, not in the BraneScript workflow).
- If the user mentions a node, site, or location name, always include a sub-task:
  "Route execution to node <name> using on attribute #[on]"

USER INTENT:
{intent}

SUB-TASKS:"""

# ---------------------------------------------------------------------------
# Prompt 2: Query rewriter: translates English sub-tasks into BraneScript-vocabulary search terms.
# ---------------------------------------------------------------------------


class IntentDecomposer:
    """
    Breaks the user's intent into a list of concrete BraneScript sub-tasks.

    Usage:
        decomposer = IntentDecomposer(llm)
        subtasks = decomposer.decompose("I want to analyze heart-disease data")
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm

        self.decompose_chain = (
            PromptTemplate.from_template(DECOMPOSE_TEMPLATE)
            | llm
            | StrOutputParser()
        )


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_subtasks(self, raw: str) -> list[str]:
        # Strip Qwen3 <think>...</think> blocks before parsing
        raw = strip_thinking_tokens(raw)

        subtasks = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.",
                           "1)", "2)", "3)", "4)", "5)", "6)", "7)", "8)", "9)", "10)", "11)", "12)",
                           "-", "*"]:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            if line:
                subtasks.append(line)

        # Filter runs ONCE after all lines are collected, not inside the loop
        subtasks = [s for s in subtasks if len(s) < 80 and not s.endswith(":")]
        return subtasks[:12]

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