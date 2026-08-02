"""
semantic_cache.py

Semantic cache for BraneScript generation results.

When a user submits an intent, the cache checks whether a semantically
identical intent has been executed before (cosine similarity >= threshold).
If so, the previous result is returned without re-running the LLM or Brane.

This brings reproducibility to scientific computing: the same analysis
produces the same result regardless of how the intent is phrased.

Storage: a dedicated ChromaDB collection ("intent_cache") in the same DB
directory as the package knowledge base.

Usage:
    from src.semantic_cache import SemanticCache

    cache = SemanticCache()

    # Lookup before generating
    hit = cache.lookup("compute glucose statistics on heal_pa_2")
    if hit:
        print("Cache hit!", hit["branescript"])
    else:
        # ... run LLM + Brane ...
        cache.store(intent, branescript, execution_result)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_BASE_DIR     = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = _BASE_DIR / "brane_pkg_db"          # same dir as package DB
CACHE_COLLECTION = "intent_cache"
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine similarity threshold — intents with similarity >= this are considered
# semantically equivalent.  0.92 is tight (near-paraphrase only); lower values
# cache more aggressively but risk false positives.
DEFAULT_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------
class SemanticCache:
    """
    Persistent semantic cache backed by ChromaDB.

    Each entry stores:
        intent          original intent text
        branescript     generated BraneScript code
        execution       JSON-encoded execution result dict (stdout, exit_code, …)
        job_id          UUID of the original job
        timestamp       ISO-8601 creation time
        hits            number of times this entry was returned as a cache hit

    Parameters
    ----------
    threshold : float
        Cosine similarity threshold in [0, 1].  Entries with similarity >=
        threshold are considered cache hits.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._ef = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL, trust_remote_code=True
        )
        self._client = chromadb.PersistentClient(path=str(CACHE_DB_PATH))
        self._col = self._client.get_or_create_collection(
            name=CACHE_COLLECTION,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, intent: str) -> Optional[dict]:
        """
        Return the best matching cached result for *intent*, or None if no
        entry meets the similarity threshold.

        The returned dict contains:
            intent          matched intent text
            branescript     generated BraneScript
            execution       dict with stdout / exit_code / success / …
            similarity      cosine similarity score (0-1)
            job_id          original job UUID
            timestamp       when the original job ran
            hits            how many times this entry has been returned
        """
        if self._col.count() == 0:
            return None

        results = self._col.query(
            query_texts=[intent],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            return None

        distance   = results["distances"][0][0]    # cosine distance in [0, 2]
        similarity = 1.0 - distance / 2.0          # convert to [0, 1]

        if similarity < self.threshold:
            return None

        meta   = results["metadatas"][0][0]
        entry_id = results["ids"][0][0]

        # Increment hit counter
        self._col.update(
            ids=[entry_id],
            metadatas=[{**meta, "hits": meta.get("hits", 0) + 1}],
        )

        execution = {}
        if meta.get("execution_json"):
            try:
                execution = json.loads(meta["execution_json"])
            except Exception:
                pass

        return {
            "intent":       results["documents"][0][0],
            "branescript":  meta.get("branescript", ""),
            "execution":    execution,
            "similarity":   round(similarity, 4),
            "job_id":       meta.get("job_id", ""),
            "timestamp":    meta.get("timestamp", ""),
            "hits":         meta.get("hits", 0) + 1,
        }

    def store(
        self,
        intent: str,
        branescript: str,
        execution: dict | None = None,
        job_id: str | None = None,
    ) -> str:
        """
        Store a new cache entry.  Returns the assigned entry ID.

        Parameters
        ----------
        intent       : natural-language intent text
        branescript  : generated BraneScript code
        execution    : dict with execution result (stdout, exit_code, …)
        job_id       : UUID of the original job (optional)
        """
        entry_id = str(uuid.uuid4())
        meta: dict = {
            "branescript":    branescript,
            "execution_json": json.dumps(execution or {}),
            "job_id":         job_id or "",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "hits":           0,
        }
        self._col.add(
            ids=[entry_id],
            documents=[intent],
            metadatas=[meta],
        )
        return entry_id

    def invalidate(self, intent: str) -> int:
        """
        Remove all cache entries whose similarity to *intent* >= threshold.
        Returns number of entries removed.
        """
        if self._col.count() == 0:
            return 0

        results = self._col.query(
            query_texts=[intent],
            n_results=min(10, self._col.count()),
            include=["distances"],
        )

        to_delete = [
            eid
            for eid, dist in zip(
                results["ids"][0], results["distances"][0]
            )
            if (1.0 - dist / 2.0) >= self.threshold
        ]
        if to_delete:
            self._col.delete(ids=to_delete)
        return len(to_delete)

    def stats(self) -> dict:
        """Return basic stats about the cache."""
        count = self._col.count()
        if count == 0:
            return {"total_entries": 0, "total_hits": 0, "threshold": self.threshold}

        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        total_hits = sum(m.get("hits", 0) for m in all_meta)
        return {
            "total_entries": count,
            "total_hits":    total_hits,
            "threshold":     self.threshold,
        }

    def list_entries(self, limit: int = 50) -> list[dict]:
        """Return up to *limit* cache entries (newest first)."""
        if self._col.count() == 0:
            return []
        all_data = self._col.get(include=["documents", "metadatas"])
        entries = []
        for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
            execution = {}
            if meta.get("execution_json"):
                try:
                    execution = json.loads(meta["execution_json"])
                except Exception:
                    pass
            entries.append({
                "intent":      doc,
                "branescript": meta.get("branescript", ""),
                "execution":   execution,
                "job_id":      meta.get("job_id", ""),
                "timestamp":   meta.get("timestamp", ""),
                "hits":        meta.get("hits", 0),
            })
        entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return entries[:limit]
