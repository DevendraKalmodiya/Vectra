import time
import numpy as np
from typing import Dict, List, Any, Optional
from src.core.base import BaseVectorIndex, SearchResult, IndexStats
from src.core.distance import l2_normalize, batch_cosine_similarity, top_k_selection
from src.core.kmeans import KMeansCluster

class IVFFlatIndex(BaseVectorIndex):
    """
    Inverted File Index (IVF-Flat) Engine.
    Partitions vector space into nlist Voronoi cells via K-Means clustering.
    Searches only the top nprobe nearest cells to trade Recall@K for speed.
    """

    def __init__(self, dimension: int = 384, nlist: int = 100, nprobe: int = 4, seed: int = 42):
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        self.kmeans = KMeansCluster(nlist=self.nlist, seed=self.seed)
        
        self.vectors: Optional[np.ndarray] = None          # Contiguous (N, D) float32 matrix
        self.inverted_lists: Dict[int, List[int]] = {}     # Maps centroid_id -> list of array offsets
        self.id_to_offset: Dict[int, int] = {}             # Maps external ID -> array offset
        self.offset_to_id: Dict[int, int] = {}             # Maps array offset -> external ID
        self.tombstones: np.ndarray = np.array([], dtype=bool)  # Soft delete bitmask
        self.next_offset: int = 0
        self.is_trained: bool = False

    def build(self, vectors: np.ndarray) -> None:
        """Trains K-Means centroids and populates cluster inverted lists."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors shape must be (N, {self.dimension})")

        norm_vectors = l2_normalize(vectors.astype(np.float32))
        n = len(norm_vectors)

        # 1. Fit K-Means centroids
        centroids, assignments = self.kmeans.fit(norm_vectors)
        self.is_trained = True

        # 2. Store vectors in contiguous memory array
        self.vectors = np.ascontiguousarray(norm_vectors, dtype=np.float32)
        self.tombstones = np.zeros(n, dtype=bool)

        # 3. Populate mappings and inverted lists
        self.id_to_offset = {i: i for i in range(n)}
        self.offset_to_id = {i: i for i in range(n)}
        self.next_offset = n

        self.inverted_lists = {c: [] for c in range(len(centroids))}
        for offset, cluster_id in enumerate(assignments):
            self.inverted_lists[cluster_id].append(offset)

    def search(self, query: np.ndarray, top_k: int = 10, **kwargs: Any) -> SearchResult:
        """Searches candidate vectors residing in the top nprobe nearest Voronoi cells."""
        if not self.is_trained or self.vectors is None or len(self.vectors) == 0:
            return SearchResult(ids=[], scores=[], latency_ms=0.0, candidates_searched=0, distance_calcs=0)

        start_time = time.perf_counter()
        nprobe = kwargs.get("nprobe", self.nprobe)
        norm_query = l2_normalize(query.astype(np.float32))

        # 1. Identify top nprobe target cluster centroids
        target_centroids = self.kmeans.predict_centroids(norm_query, nprobe=nprobe)

        # 2. Gather candidate offsets across target inverted lists
        candidate_offsets: List[int] = []
        for c_id in target_centroids:
            candidate_offsets.extend(self.inverted_lists.get(c_id, []))

        if not candidate_offsets:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(ids=[], scores=[], latency_ms=elapsed_ms, candidates_searched=0, distance_calcs=0)

        candidate_offsets_arr = np.array(candidate_offsets, dtype=np.int64)

        # 3. Filter out soft-deleted tombstones
        active_mask = ~self.tombstones[candidate_offsets_arr]
        active_offsets = candidate_offsets_arr[active_mask]

        if len(active_offsets) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return SearchResult(ids=[], scores=[], latency_ms=elapsed_ms, candidates_searched=len(candidate_offsets_arr), distance_calcs=0)

        # 4. Compute inner-product similarity against candidate subset
        candidate_matrix = self.vectors[active_offsets]
        candidate_scores = batch_cosine_similarity(norm_query, candidate_matrix)

        # 5. Extract top_k items
        k = min(top_k, len(candidate_scores))
        top_sub_idx, top_scores = top_k_selection(candidate_scores, k)
        
        final_offsets = active_offsets[top_sub_idx]
        final_ids = [self.offset_to_id[off] for off in final_offsets]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return SearchResult(
            ids=final_ids,
            scores=top_scores.tolist(),
            latency_ms=elapsed_ms,
            candidates_searched=len(candidate_offsets_arr),
            distance_calcs=len(active_offsets)
        )

    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        """Inserts a new vector into the nearest centroid inverted list."""
        if vector.shape != (self.dimension,):
            raise ValueError(f"Vector dimension must be ({self.dimension},)")

        norm_vec = l2_normalize(vector.astype(np.float32)).reshape(1, self.dimension)

        if not self.is_trained or self.vectors is None:
            # Fallback initialization if inserted before batch training
            self.vectors = norm_vec
            self.tombstones = np.array([False], dtype=bool)
            self.kmeans.centroids = norm_vec
            self.inverted_lists = {0: [0]}
            self.is_trained = True
            cluster_id = 0
        else:
            self.vectors = np.vstack((self.vectors, norm_vec))
            self.tombstones = np.append(self.tombstones, False)
            cluster_id = int(self.kmeans.predict_centroids(norm_vec[0], nprobe=1)[0])
            self.inverted_lists.setdefault(cluster_id, []).append(self.next_offset)

        offset = self.next_offset
        self.id_to_offset[vector_id] = offset
        self.offset_to_id[offset] = vector_id
        self.next_offset += 1

    def delete(self, vector_id: int) -> bool:
        """Soft-deletes a vector ID by marking its tombstone bitmask."""
        if vector_id not in self.id_to_offset:
            return False
        offset = self.id_to_offset[vector_id]
        if self.tombstones[offset]:
            return False
        self.tombstones[offset] = True
        return True

    def get_stats(self) -> IndexStats:
        total = len(self.vectors) if self.vectors is not None else 0
        deleted = int(np.sum(self.tombstones)) if len(self.tombstones) > 0 else 0
        return IndexStats(
            total_vectors=total,
            dimension=self.dimension,
            deleted_vectors=deleted,
            index_type=f"IVFFlat(nlist={self.nlist}, nprobe={self.nprobe})",
            is_trained=self.is_trained
        )