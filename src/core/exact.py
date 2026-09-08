import time
import numpy as np
from typing import Dict, Any, Optional
from src.core.base import BaseVectorIndex, SearchResult, IndexStats
from src.core.distance import l2_normalize, batch_cosine_similarity, top_k_selection

class ExactIndex(BaseVectorIndex):
    """
    Brute-force O(N * D) cosine search engine.
    Computes exact dot products over contiguous float32 memory to generate ground truth.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: Optional[np.ndarray] = None  # Contiguous (N, D) array
        self.id_to_offset: Dict[int, int] = {}     # Maps external ID -> array offset
        self.offset_to_id: Dict[int, int] = {}     # Maps array offset -> external ID
        self.tombstones: np.ndarray = np.array([], dtype=bool)  # Soft delete bitmask
        self.next_offset: int = 0

    def build(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors must have shape (N, {self.dimension})")
        
        normalized_vectors = l2_normalize(vectors.astype(np.float32))
        n = len(normalized_vectors)
        self.vectors = np.ascontiguousarray(normalized_vectors, dtype=np.float32)
        self.tombstones = np.zeros(n, dtype=bool)
        
        self.id_to_offset = {i: i for i in range(n)}
        self.offset_to_id = {i: i for i in range(n)}
        self.next_offset = n

    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension must be ({self.dimension},)")
        
        norm_vec = l2_normalize(vector.astype(np.float32)).reshape(1, self.dimension)
        
        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = norm_vec
            self.tombstones = np.array([False], dtype=bool)
        else:
            self.vectors = np.vstack([self.vectors, norm_vec])
            self.tombstones = np.append(self.tombstones, False)
        
        offset = self.next_offset
        self.id_to_offset[vector_id] = offset
        self.offset_to_id[offset] = vector_id
        self.next_offset += 1

    def search(self, query: np.ndarray, top_k: int = 10, **kwargs: Any) -> SearchResult:
        if self.vectors is None or len(self.vectors) == 0:
            return SearchResult(ids=[], scores=[], latency_ms=0.0, candidates_searched=0, distance_calcs=0)

        start_time = time.perf_counter()
        norm_query = l2_normalize(query.astype(np.float32))
        
        # O(N * D) Matrix Dot Product
        raw_scores = batch_cosine_similarity(norm_query, self.vectors)
        
        # Mask out soft-deleted IDs
        if np.any(self.tombstones):
            raw_scores = raw_scores.copy()
            raw_scores[self.tombstones] = -np.inf

        active_candidates = int(np.sum(~self.tombstones))
        k = min(top_k, active_candidates)

        if k <= 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(ids=[], scores=[], latency_ms=elapsed_ms, candidates_searched=0, distance_calcs=0)

        top_offsets, top_scores = top_k_selection(raw_scores, k)
        top_ids = [self.offset_to_id[off] for off in top_offsets]
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            ids=top_ids,
            scores=top_scores.tolist(),
            latency_ms=elapsed_ms,
            candidates_searched=len(self.vectors),
            distance_calcs=len(self.vectors)
        )

    def delete(self, vector_id: int) -> bool:
        if vector_id not in self.id_to_offset:
            return False
        offset = self.id_to_offset[vector_id]
        if self.tombstones[offset]:
            return False  # Already soft-deleted
        self.tombstones[offset] = True
        return True

    def get_stats(self) -> IndexStats:
        total = len(self.vectors) if self.vectors is not None else 0
        deleted = int(np.sum(self.tombstones)) if len(self.tombstones) > 0 else 0
        return IndexStats(
            total_vectors=total,
            dimension=self.dimension,
            deleted_vectors=deleted,
            index_type="ExactIndex",
            is_trained=True
        )