import json
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from sentence_transformers import SentenceTransformer
from src.core.exact import ExactIndex
from src.core.ivf_flat import IVFFlatIndex
from src.storage.tombstone import TombstoneManager
from src.core.base import SearchResult, IndexStats
import config

class SearchService:
    """
    Service layer orchestrating Exact and IVF-Flat search engines,
    on-the-fly text embeddings, text-based document lookups, tombstone soft-deletions, and compaction.
    """

    def __init__(self):
        self.dimension = config.VECTOR_DIM
        self.exact_index = ExactIndex(dimension=self.dimension)
        self.ivf_index = IVFFlatIndex(dimension=self.dimension, nlist=config.DEFAULT_NLIST, seed=config.RANDOM_SEED)
        self.tombstone_manager = TombstoneManager()
        self.model: Optional[SentenceTransformer] = None
        self.text_store: Dict[int, str] = {}

    def initialize(self, load_data: bool = True) -> None:
        """Loads precomputed vectors and text store, and builds index instances."""
        if load_data:
            if config.VECTOR_STORE_PATH.exists():
                vectors = np.load(config.VECTOR_STORE_PATH)
                self.exact_index.build(vectors)
                self.ivf_index.build(vectors)
            if config.TEXT_STORE_PATH.exists():
                with open(config.TEXT_STORE_PATH, "r", encoding="utf-8") as f:
                    raw_texts = json.load(f)
                    self.text_store = {i: txt for i, txt in enumerate(raw_texts)}

    def _get_model(self) -> SentenceTransformer:
        """Lazy-loads the embedding transformer model."""
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def get_text(self, vector_id: int) -> str:
        """Retrieves stored text for a vector ID, or fallback placeholder string."""
        return self.text_store.get(vector_id, f"Vector ID #{vector_id}")

    def find_documents(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Searches active documents by semantic meaning, identifies exact string matches,
        and returns candidate documents for explicit user deletion selection.
        """
        if not query_text or not query_text.strip():
            return []

        model = self._get_model()
        query_vector = model.encode(query_text, convert_to_numpy=True).astype(np.float32)
        
        search_res = self.exact_index.search(query_vector, top_k=top_k)

        matches = []
        normalized_query = query_text.strip().lower()

        for doc_id, score in zip(search_res.ids, search_res.scores):
            doc_text = self.get_text(doc_id)
            is_exact = (doc_text.strip().lower() == normalized_query)
            matches.append({
                "id": doc_id,
                "text": doc_text,
                "score": float(score),
                "is_exact_match": is_exact
            })

        # Ensure exact string matches sort to the top
        matches.sort(key=lambda m: (not m["is_exact_match"], -m["score"]))
        return matches

    def search(
        self,
        query_vector: Optional[np.ndarray],
        query_text: Optional[str],
        index_type: str,
        top_k: int,
        nprobe: int
    ) -> SearchResult:
        if query_vector is None and query_text is not None:
            model = self._get_model()
            query_vector = model.encode(query_text, convert_to_numpy=True).astype(np.float32)

        if query_vector is None:
            raise ValueError("Either query_vector or query_text must be provided.")

        if index_type.lower() == "exact":
            return self.exact_index.search(query_vector, top_k=top_k)
        else:
            return self.ivf_index.search(query_vector, top_k=top_k, nprobe=nprobe)

    def insert(
        self,
        vector_id: int,
        vector: Optional[np.ndarray] = None,
        text: Optional[str] = None
    ) -> np.ndarray:
        """Inserts a raw vector directly or embeds text on-the-fly before inserting."""
        if vector is None and text is not None:
            model = self._get_model()
            vector = model.encode(text, convert_to_numpy=True).astype(np.float32)
        if vector is None:
            raise ValueError("Must provide either a vector or text.")

        if text is not None:
            self.text_store[vector_id] = text

        self.exact_index.insert(vector_id, vector)
        self.ivf_index.insert(vector_id, vector)
        return vector

    def delete(self, vector_id: int) -> bool:
        """Soft-deletes a vector ID across both indices using tombstones."""
        d1 = self.exact_index.delete(vector_id)
        d2 = self.ivf_index.delete(vector_id)
        if d1 or d2:
            self.tombstone_manager.mark_deleted(vector_id)
            return True
        return False

    def compact(self) -> Tuple[int, int]:
        """Purges soft-deleted tombstone entries and reclaims memory across indices."""
        purged_exact = self.tombstone_manager.compact_index(self.exact_index)
        purged_ivf = self.tombstone_manager.compact_index(self.ivf_index)
        return purged_exact, purged_ivf

    def get_cluster_sizes(self) -> Dict[int, int]:
        """Retrieves vector density distribution across all IVF clusters."""
        return self.ivf_index.get_cluster_sizes()

    def get_vector_cluster(self, vector_id: int) -> int:
        """Retrieves the assigned cluster ID for a given vector ID."""
        return self.ivf_index.get_vector_cluster(vector_id)

    def get_stats(self) -> Dict[str, IndexStats]:
        return {
            "exact": self.exact_index.get_stats(),
            "ivf": self.ivf_index.get_stats()
        }

search_service = SearchService()