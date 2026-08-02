"""
pkg_retriever.py

PACKAGE / DATASET RETRIEVAL (context-relevant DB)

Two-stage retrieval:
  1. Name-pinned: if the intent mentions a known package or dataset name OR a
     known keyword alias for that package, retrieve ALL docs for that package/
     dataset via metadata filter (guaranteed complete API coverage).
  2. Similarity fallback: when no specific names/aliases are detected, fall
     back to embedding similarity search across all packages/datasets.

This scales to any number of packages — still one DB, constant query time.
Adding a new package only requires:
  1. Adding a keywords.txt file in the package folder.
  2. Rebuilding the DB (python src/knowledgeBase.py).
No changes to this file are needed.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List, Optional


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
        self.package_aliases: dict[str, list[str]] = {}  # populated by _load_known_names
        self.known_packages, self.known_datasets = self._load_known_names()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_known_names(self) -> tuple[set[str], set[str]]:
        """
        Scan DB metadata once at startup to discover all known package/dataset names
        and load keyword aliases from package_keywords documents.
        """
        result = self.pkg_db.get(include=["metadatas"])
        packages: set[str] = set()
        datasets: set[str] = set()
        aliases: dict[str, list[str]] = {}

        for meta in (result.get("metadatas") or []):
            if not meta:
                continue
            if "package" in meta:
                packages.add(meta["package"])
            if "dataset" in meta:
                datasets.add(meta["dataset"])
            # Load aliases stored by knowledgeBase.py from keywords.txt
            if meta.get("doc_type") == "package_keywords" and "keywords" in meta:
                pkg = meta["package"]
                kws = [k.strip() for k in meta["keywords"].split(",") if k.strip()]
                aliases.setdefault(pkg, [])
                aliases[pkg].extend(kws)

        self.package_aliases = aliases
        print(f"📚 Known packages: {sorted(packages)}")
        print(f"📚 Known datasets: {sorted(datasets)}")
        print(f"📚 Loaded aliases for: {sorted(aliases.keys())}")
        return packages, datasets

    def _detect_names(self, text: str) -> tuple[set[str], set[str]]:
        """Scan text for known package/dataset names and keyword aliases (case-insensitive)."""
        text_lower = text.lower()
        found_pkgs: set[str] = set()
        found_ds: set[str] = set()

        # Direct name match
        for p in self.known_packages:
            if p.lower() in text_lower:
                found_pkgs.add(p)

        for d in self.known_datasets:
            if d.lower() in text_lower:
                found_ds.add(d)

        # Alias match — resolve domain terms to package names
        for pkg, aliases in self.package_aliases.items():
            if pkg in self.known_packages:
                for alias in aliases:
                    if alias in text_lower:
                        found_pkgs.add(pkg)
                        break

        return found_pkgs, found_ds

    def _get_by_metadata(self, field: str, value: str) -> List[Document]:
        """Retrieve all docs for a specific package or dataset via metadata filter."""
        result = self.pkg_db.get(
            where={field: value},
            include=["documents", "metadatas"],
        )
        docs = []
        for content, meta in zip(result.get("documents") or [], result.get("metadatas") or []):
            if content:
                docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def _similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
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

        Stage 1 — Name-pinned: scan the intent+subtasks for known package/dataset
          names and retrieve ALL docs for each match via metadata filter.
          This guarantees the correct API is always present for named packages/datasets.
        Stage 2 — Similarity fallback: when no specific names are detected,
          fall back to embedding similarity search (handles vague/ambiguous intents).
        """
        print("\n📦 Package/dataset retrieval:")

        full_text = user_query + " " + " ".join(subtasks)
        found_pkgs, found_ds = self._detect_names(full_text)

        all_docs: List[Document] = []

        for pkg in sorted(found_pkgs):
            print(f"   📌 Pinned package: {pkg}")
            all_docs.extend(self._get_by_metadata("package", pkg))

        for ds in sorted(found_ds):
            print(f"   📌 Pinned dataset: {ds}")
            all_docs.extend(self._get_by_metadata("dataset", ds))

        if not found_pkgs and not found_ds:
            print("   🔍 No names detected — falling back to similarity search")
            # Use larger k so we cover multiple packages
            all_docs.extend(self._similarity_search(user_query, k=8))
            for st in subtasks:
                all_docs.extend(self._similarity_search(st, k=3))

        docs = self._deduplicate(all_docs)
        print(f"   → {len(docs)} unique chunks returned")

        if not docs:
            return "(No package/dataset documentation found for this query.)"

        return "\n\n---\n\n".join(doc.page_content for doc in docs)
