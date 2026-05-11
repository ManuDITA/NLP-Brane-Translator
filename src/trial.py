"""
trial.py  (updated — integrates IntentDecomposer)

Changes from the original:
  - IntentDecomposer replaces the plain db.as_retriever() call
  - The RAG chain now receives pre-expanded, deduplicated context
  - Sub-tasks are injected into the prompt so the LLM knows the structure
"""

import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from intent_decomposer import IntentDecomposer   # <-- new import

# --- CONFIGURATION ---
CHROMA_PATH = "../brane_knowledge_db"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3", temperature=0)

# 1. Load the database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# 2. Build the decomposer  (reuses same llm + db — no extra overhead)
decomposer = IntentDecomposer(llm=llm, db=db, k_per_subtask=3)

# 3. Define the prompt
#    Now includes {subtasks} so the LLM knows the expected structure.
template = """
You are an expert for the Brane Framework and BraneScript.

The user's request has been broken into the following sub-tasks.
Address EVERY sub-task in order in your output.

SUB-TASKS:
{subtasks}

Use the documentation below to write correct BraneScript.
Output ONLY valid BraneScript code. No explanations outside of inline comments.
Use comments (// ...) to mark assumptions about parameters or dataset names.

DOCUMENTATION CONTEXT:
{context}

USER REQUEST:
{question}

BRANESCRIPT CODE:
"""

prompt = ChatPromptTemplate.from_template(template)

# 4. Generation chain (context + subtasks come from decomposer, not a retriever)
generation_chain = prompt | llm | StrOutputParser()


def run_pipeline(user_query: str) -> str:
    # Step A: decompose intent + retrieve expanded context
    context, subtasks = decomposer.run(user_query)

    subtasks_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(subtasks))

    # Step B: generate BraneScript
    result = generation_chain.invoke({
        "context": context,
        "subtasks": subtasks_str,
        "question": user_query,
    })
    return result


# --- EXECUTION ---
if __name__ == "__main__":
    user_query = (
        "I want to run a private analysis on the heart-disease dataset using package A. "
        "Make sure to use the correct BraneScript syntax for function definition and package usage. "
        "If you need to make assumptions about the input parameters, do so and clearly indicate "
        "them in comments within the BraneScript code. "
        "Generate additional stuff that is needed to make the code work, such as imports or "
        "helper functions. The function should be self-contained and executable within the "
        "BraneScript environment, given the appropriate input parameters."
    )

    print(f"🧠 Intent: {user_query}\n")
    print("⏳ Running decomposed RAG pipeline...")

    result = run_pipeline(user_query)

    print("\n" + "=" * 50)
    print("FINAL BRANESCRIPT OUTPUT:")
    print("=" * 50)
    print(result)