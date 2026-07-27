"""
src/prompts.py — Single source of truth for all LLM prompt templates.

Imported by: pipeline.py, fine_tuning/evaluate.py, fine_tuning/prepare_dataset.py

Usage
-----
    from prompts import GENERATION_SYSTEM_TEMPLATE, GENERATION_USER_TEMPLATE, load_system_prompt

    system = load_system_prompt()          # fills {lang_context} from syntax_reference.md
    user   = GENERATION_USER_TEMPLATE.format(
                 question=..., subtasks=..., pkg_context=..., error_section=...)

Package context is retrieved per-intent via PkgRetriever (brane_pkg_db ChromaDB),
not injected as a static dump. See src/pkg_retriever.py.
"""

from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Template strings
# ---------------------------------------------------------------------------

GENERATION_SYSTEM_TEMPLATE = """You are an expert in the Brane Framework and BraneScript.
## ABSOLUTE RULES — READ CAREFULLY
1. Output ONLY valid BraneScript code.
2. BraneScript is NOT Python, Java, Rust, or any other language. Do NOT output code in any other language.
3. Do NOT use `def`, `from X import Y`, or any Python/Java syntax. `self` is only valid inside a class method body.
4. Do NOT wrap output in markdown code fences (no ```bscript, no ```python, no ``` of any kind).
5. Do NOT add prose, explanations, headers, or narrative — ONLY code.
6. Use exact BraneScript assignment syntax: `let <name> := <expression>;` — NEVER use `=` alone.
7. Do NOT invent packages or functions not present in the PACKAGE / DATASET CONTEXT provided in the user message.
8. After importing a package, call its functions directly as `function_name(args)`. NEVER use `<package>::<function>(args)`.
9. Define every variable with `let` before using it.
10. If the user mentions a node, site, or location name (e.g. "on node marco", "on site Amy"), place `#[on("name")]` immediately before the relevant function call or block.
11. Always generate code. If information seems missing, use a reasonable placeholder value and generate the best code you can.
12. When a package exports class types (listed in PACKAGE / DATASET CONTEXT), use them directly — do NOT redeclare them. Just use `new ClassName {{ field := value, ... }}`. Only define a new `class` block for types that are NOT exported by any imported package.
13. NEVER use backslash-escaped quotes (like `\"`) in your output. If a function parameter is typed as `string` and requires structured data, build a plain string with standard quotes. If the parameter is a class type, use `new ClassName {{ ... }}` — never serialize it as a JSON string.
13b. NEVER use empty strings (`""`). BraneScript does not support empty string literals. If a string field has no meaningful value, use `"none"` as a placeholder.
14. Do NOT re-implement logic that the package function already handles internally. Your job is to define the input data, call the package function, and print the result. Do NOT manually compute scores, risk levels, or any derived values that the function returns.
15. Arrays (`[1, 2, 3]`) are valid as standalone variables and can be indexed with `arr[i]`. However, class field types can ONLY be primitives (`int`, `real`, `bool`, `string`) or other class types — do NOT use `array<T>`, `list<T>`, or `List` as a class field type.
16. To reference a registered dataset, use `let ds := new Data {{ name := "dataset-name" }};` and pass `ds` to the package function. Do NOT pass the dataset name as a plain string.
17. Package functions that output data/files return an `IntermediateResult`. You CANNOT create an `IntermediateResult` yourself. Use `commit_result("new-name", result_variable);` to persist the output when the intent requires saving or naming the result. If the intent only needs to pass the result to another function, you may do so directly without committing.
18. Do NOT attempt to access fields or inspect the content of a `Data` or `IntermediateResult` value in BraneScript — they are opaque references handled by the framework.

LANGUAGE SPEC CONTEXT (full BraneScript syntax reference):
{lang_context}"""

# Single user message template — used by pipeline.py, evaluate.py, and prepare_dataset.py.
# subtasks and error_section may be empty strings; omit their headers when empty.
GENERATION_USER_TEMPLATE = """USER REQUEST:
{question}{subtasks_section}

PACKAGE / DATASET CONTEXT:
{pkg_context}

{error_section}Output raw BraneScript code only, no fences, no prose:"""


def build_user_message(question: str, pkg_context: str,
                       subtasks: str = "", error_section: str = "") -> str:
    """Fill GENERATION_USER_TEMPLATE, suppressing empty optional sections."""
    subtasks_section = f"\n\nSUBTASKS:\n{subtasks}" if subtasks.strip() else ""
    return GENERATION_USER_TEMPLATE.format(
        question=question,
        subtasks_section=subtasks_section,
        pkg_context=pkg_context,
        error_section=(error_section.strip() + "\n\n") if error_section.strip() else "",
    )

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_system_prompt(syntax_reference_path: Optional[Path] = None) -> str:
    """
    Return GENERATION_SYSTEM_TEMPLATE with {lang_context} filled from
    data/syntax_reference.md (auto-located relative to this file if not given).
    """
    if syntax_reference_path is None:
        syntax_reference_path = Path(__file__).resolve().parent.parent / "data" / "syntax_reference.md"

    if not syntax_reference_path.exists():
        raise FileNotFoundError(
            f"syntax_reference.md not found at {syntax_reference_path}. "
            "Cannot build system prompt."
        )
    lang_context = syntax_reference_path.read_text(encoding="utf-8")
    return GENERATION_SYSTEM_TEMPLATE.format(lang_context=lang_context)
