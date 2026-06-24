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
from typing import List, Optional

# Maximum unique chunks returned to the prompt — keeps context window bounded.
MAX_PKG_CHUNKS = 8

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

    def _retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        return self.pkg_db.similarity_search(query, k=k or self.k)

    def _deduplicate(self, docs: List[Document]) -> List[Document]:
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

    def run(self, subtasks: List[str], user_query: str) -> str:
        """
        Retrieve package/dataset context based on the user's intent.

        Queries with both the full user intent (broad signal) and each
        subtask (precise signal), deduplicates, and caps at MAX_PKG_CHUNKS.
        """
        print("\n📦 Package/dataset retrieval:")

        all_docs: list[Document] = []

        # Broad query — full user intent gives the best semantic match
        print(f"   • intent: {user_query}")
        all_docs.extend(self._retrieve(user_query, k=self.k))

        # Precise queries — subtasks target specific function/package references
        # Use k=2 per subtask to avoid flooding the context window
        for st in subtasks:
            all_docs.extend(self._retrieve(st, k=2))

        docs = self._deduplicate(all_docs)[:MAX_PKG_CHUNKS]
        print(f"   → {len(docs)} unique chunks returned")

        if not docs:
            return " (No package/dataset documentation found for this query.)"

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
