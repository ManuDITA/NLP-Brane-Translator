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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, subtasks: list[str], user_query: str) -> str:
        """
        Retrieve package/dataset context based on the user's intent.

        Uses the full user query to find the most relevant package content.
        Returns the package context as a single block of text.
        """
        print("\n📦 Package/dataset retrieval:")
        print(f"   • intent: {user_query}")

        docs = self._deduplicate(self._retrieve(user_query))
        print(f"   → {len(docs)} unique chunks returned")

        if not docs:
            return " (No package/dataset documentation found for this query.)"

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
