"""
example_generator.py

Generates additional (intent, BraneScript) example pairs using the locally
hosted HuggingFace LLM and appends them to data/examples/generated.jsonl.

The script asks the model to produce BraneScript for a set of seed intents,
then saves valid outputs as new training examples.

Run (on Snellius, with the venv active):
    python src/example_generator.py [--count N] [--category CATEGORY] [--model MODEL_ID]

Options:
    --count     Number of examples to generate (default: 20)
    --category  Target category: basic | control_flow | functions |
                classes_arrays | healthcare | advanced | all (default: all)
    --model     HuggingFace model ID (default: Qwen/Qwen3-4B)
    --temperature  Sampling temperature (default: 0.5)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseLanguageModel

from utils import (
    strip_thinking_tokens,
    strip_code_fences,
    looks_like_branescript,
    detect_json_string_assignment,
    load_hf_token,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = BASE_DIR / "data" / "examples"
GENERATED_FILE = EXAMPLES_DIR / "generated.jsonl"

# ---------------------------------------------------------------------------
# Seed intents by category
# ---------------------------------------------------------------------------
SEED_INTENTS = {
    "basic": [
        "Declare a variable for a patient's name and print it",
        "Add two numbers and print the result",
        "Declare a boolean flag and use it in an expression",
        "Swap the values of two integer variables",
        "Concatenate three strings into a single message and print it",
    ],
    "control_flow": [
        "Print even numbers from 0 to 20 using a for loop",
        "Loop while a counter is below 10 and double it each iteration",
        "Check if a patient's age is above 60 and print a warning",
        "Iterate through indices 0 to 4 and print only the ones divisible by 2",
        "Count down from 10 to 1 using a while loop",
    ],
    "functions": [
        "Define a function that converts Celsius to Fahrenheit and test it",
        "Define a function that returns the maximum of two integers",
        "Define a recursive function to compute the Fibonacci number at index n",
        "Define a function that counts characters in a string and returns the count",
        "Define a function that builds a formatted summary string from individual fields",
    ],
    "classes_arrays": [
        "Define a class for a hospital with a name and bed count, create an instance",
        "Create an array of five integers and compute their sum with a loop",
        "Define a class with a method that returns a formatted description string",
        "Iterate over an array of patient IDs and print each one",
        "Create a 2D array representing a 3x3 grid and print the center element",
    ],
    "healthcare": [
        "Analyze heart disease risk for a 40-year-old female with normal blood pressure",
        "Generate a health report for a patient with smoking history aged 60",
        "Validate patient data for a 25-year-old male before analysis",
        "Analyze heart disease risk and then generate a full report for the same patient",
        "Run heart disease analysis for a diabetic 70-year-old female patient",
    ],
    "advanced": [
        "Run two independent analyses in parallel and print both results",
        "Use a dataset reference and pass it to a function for processing",
        "Tag a package function call to run on a specific site named Bob",
        "Run parallel analyses on the same dataset from two different sites",
        "Define a function, call it in a parallel block, and collect results",
    ],
}

# ---------------------------------------------------------------------------
# Generation prompt — {no_think_prefix} injected for Qwen3 models
# ---------------------------------------------------------------------------
GENERATION_PROMPT = """{no_think_prefix}You are an expert in BraneScript, the workflow language for the Brane Framework.

Generate ONLY valid BraneScript code for the following intent.
BraneScript rules:
- Variables: `let <name> := <value>;`
- Import: `import <package>;`
- Package calls: `function_name(args)` after importing the package
- Functions: `func name(param: type) -> type {{ ... }}`
- Builtin output: `println(value);`
- Boolean literals: `true`, `false` (lowercase)
- For structured data (e.g. patient records), define a `class` and use `new ClassName {{ field := value }}`.
  NEVER pass structured data as a JSON string with escaped quotes like `let x := "{{\\"key\\": \\"val\\"}}"`.
- No Python, no Java, no markdown fences, no prose.

INTENT: {intent}

BRANESCRIPT CODE:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_existing(filepath: Path) -> set:
    """Return set of already-generated intents to avoid duplicates."""
    existing = set()
    if not filepath.exists():
        return existing
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if "intent" in entry:
                    existing.add(entry["intent"].strip().lower())
            except json.JSONDecodeError:
                continue
    return existing


def load_model(model_id: str, temperature: float) -> BaseLanguageModel:
    """Load a HuggingFace model and wrap it as a LangChain LLM."""
    load_hf_token()
    print(f"📥 Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"📥 Loading model onto GPU (bfloat16)...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    hf_model.generation_config.max_length = 4096
    hf_model.generation_config.max_new_tokens = None

    gen_pipeline = hf_pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        return_full_text=False,
    )
    return HuggingFacePipeline(
        pipeline=gen_pipeline,
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
        },
    )


def generate_examples(llm: BaseLanguageModel, intents: List[str],
                      existing: set, no_think_prefix: str = "") -> list:
    prompt = PromptTemplate.from_template(GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    results = []
    for intent in intents:
        if intent.strip().lower() in existing:
            print(f"  ⏭  Skip (duplicate): {intent[:60]}")
            continue

        print(f"  ⚙️  Generating: {intent[:70]}")
        try:
            raw = chain.invoke({"intent": intent, "no_think_prefix": no_think_prefix})
            code = strip_thinking_tokens(raw)
            code = strip_code_fences(code)

            if looks_like_branescript(code) and not detect_json_string_assignment(code):
                results.append({"intent": intent, "branescript": code})
                print(f"      ✅ OK ({len(code)} chars)")
            elif detect_json_string_assignment(code):
                print(f"      ❌ JSON-string antipattern — skipped")
            else:
                print(f"      ❌ Output does not look like BraneScript — skipped")
        except Exception as e:
            print(f"      ⚠️  Error: {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20,
                        help="Number of examples to generate (default: 20)")
    parser.add_argument("--category", default="all",
                        choices=list(SEED_INTENTS.keys()) + ["all"],
                        help="Category of intents to use (default: all)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-4B)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature (default: 0.5)")
    args = parser.parse_args()

    print(f"🔧 Starting example generator")
    print(f"   model={args.model}  count={args.count}  category={args.category}")

    is_qwen3 = "qwen3" in args.model.lower()
    no_think_prefix = "/no_think\n" if is_qwen3 else ""
    if is_qwen3:
        print(f"ℹ️  Qwen3 detected — injecting /no_think to suppress reasoning")

    llm = load_model(args.model, args.temperature)
    print("✅ Model loaded")

    if args.category == "all":
        all_intents = []
        for cat_intents in SEED_INTENTS.values():
            all_intents.extend(cat_intents)
    else:
        all_intents = SEED_INTENTS[args.category]

    all_intents = all_intents[:args.count]

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing(GENERATED_FILE)
    print(f"📋 {len(existing)} existing generated examples")
    print(f"📝 Generating up to {len(all_intents)} new examples...")

    new_examples = generate_examples(llm, all_intents, existing,
                                     no_think_prefix=no_think_prefix)

    if new_examples:
        with open(GENERATED_FILE, "a", encoding="utf-8") as f:
            for entry in new_examples:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n✅ Appended {len(new_examples)} examples to {GENERATED_FILE}")
    else:
        print("\n⚠️  No valid examples generated.")

    print("\nNext steps:")
    print("  1. Review data/examples/generated.jsonl for quality")
    print("  2. Run: python fine_tuning/prepare_dataset.py")
    print("  3. Run: python fine_tuning/train.py")


if __name__ == "__main__":
    main()
