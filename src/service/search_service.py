import numpy as np
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.core.exact import ExactIndex
from src.core.ivf_flat import IVFFlatIndex
from src.storage.tombstone import TombstoneManager
from src.core.base import SearchResult, IndexStats
import config

class SearchService:
    """
    Service layer orchestrating Exact and IVF-Flat search engines,
    on-the-fly text embeddings, and tombstone soft-deletions.
    """

    def __init__(self):
        self.dimension = config.VECTOR_DIM
        self.exact_index = ExactIndex(dimension=self.dimension)
        self.ivf_index = IVFFlatIndex(dimension=self.dimension, nlist=config.DEFAULT_NLIST, seed=config.RANDOM_SEED)
        self.tombstone_manager = TombstoneManager()
        self.model: Optional[SentenceTransformer] = None

    def initialize(self, load_data: bool = True) -> None:
        """Loads precomputed vectors and builds index instances."""
        if load_data and config.VECTOR_STORE_PATH.exists():
            vectors = np.load(config.VECTOR_STORE_PATH)
            self.exact_index.build(vectors)
            self.ivf_index.build(vectors)

    def _get_model(self) -> SentenceTransformer:
        """Lazy-loads the embedding transformer model."""
        if self.model is None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

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

    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        self.exact_index.insert(vector_id, vector)
        self.ivf_index.insert(vector_id, vector)

    def delete(self, vector_id: int) -> bool:
        d1 = self.exact_index.delete(vector_id)
        d2 = self.ivf_index.delete(vector_id)
        if d1 or d2:
            self.tombstone_manager.mark_deleted(vector_id)
            return True
        return False

    def get_stats(self) -> Dict[str, IndexStats]:
        return {
            "exact": self.exact_index.get_stats(),
            "ivf": self.ivf_index.get_stats()
        }

search_service = SearchService()