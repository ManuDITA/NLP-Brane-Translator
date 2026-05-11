"""
pkg_retriever.py

PACKAGE / DATASET RETRIEVAL (context-relevant DB)

Takes the sub-tasks from the intent decomposer and searches the
package/dataset DB for relevant context: function signatures, input/output
types, dataset schemas, available fields.
"""

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document

MAX_PKG_CONTEXT_CHARS = 400 * 4     # ~400 tokens — pkg info is dense, keep it tight


class PkgRetriever:
    """
    Retrieves package and dataset documentation from the context-relevant DB.

    Usage:
        retriever = PkgRetriever(pkg_db)
        pkg_context = retriever.run(subtasks, user_query)
    """

    def __init__(self, pkg_db: Chroma, k: int = 4):
        self.pkg_db = pkg_db
        self.k = k

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_named_entities(self, text: str) -> list[str]:
        """
        Pull out likely package/dataset names from text.
        Heuristic: quoted strings and capitalised words are candidates.
        e.g. 'using package "Healthcare"' → ["Healthcare"]
             'heart-disease dataset'      → ["heart-disease"]
        """
        import re
        entities = []
        # Quoted strings: "Healthcare", 'heart-disease'
        entities += re.findall(r'["\']([^"\']+)["\']', text)
        # Words after "package", "dataset", "using"
        entities += re.findall(
            r'(?:package|dataset|using)\s+([A-Za-z][A-Za-z0-9_\-]*)', text,
            re.IGNORECASE
        )
        return list(dict.fromkeys(entities))   # deduplicate, preserve order

    def _retrieve(self, query: str) -> list[Document]:
        return self.pkg_db.similarity_search(query, k=self.k)

    def _deduplicate(self, docs: list[Document]) -> list[Document]:
        seen, unique = set(), []
        for doc in docs:
            key = doc.page_content[:120]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique

    def _apply_token_budget(self, docs: list[Document]) -> list[Document]:
        total, kept = 0, []
        for doc in docs:
            size = len(doc.page_content)
            if total + size > MAX_PKG_CONTEXT_CHARS:
                break
            kept.append(doc)
            total += size
        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, subtasks: list[str], user_query: str) -> str:
        """
        Retrieve package/dataset context relevant to the sub-tasks.

        Strategy:
          1. Extract named packages/datasets from user query (exact intent)
          2. Run a vector search per named entity
          3. Run a vector search per sub-task as fallback
          4. Deduplicate + apply token budget

        Returns:
            pkg_context (str): package/dataset docs to inject into the prompt
        """
        print("\n📦 Package/dataset retrieval:")
        all_docs = []

        # Step 1+2: named entity search (highest priority)
        entities = self._extract_named_entities(user_query)
        for entity in entities:
            print(f"   • exact: {entity}")
            all_docs.extend(self._retrieve(entity))

        # Step 3: sub-task fallback search
        for st in subtasks:
            if any(kw in st.lower() for kw in ["package", "dataset", "import", "data"]):
                print(f"   • sub-task: {st}")
                all_docs.extend(self._retrieve(st))

        docs = self._apply_token_budget(self._deduplicate(all_docs))
        print(f"   → {len(docs)} unique chunks within token budget")

        if not docs:
            return "(No package/dataset documentation found for this query.)"

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
