import numpy as np
from typing import Dict, Set, Any
from src.core.base import BaseVectorIndex


class TombstoneManager:
    """
    Manages soft-deletion tracking, tombstone bitmask states, and index compaction.
    Frees memory and cleans up offset mappings once deleted entries cross a threshold ratio.
    """

    def __init__(self, compaction_threshold: float = 0.2):
        self.compaction_threshold = compaction_threshold
        self.deleted_ids: Set[int] = set()

    def mark_deleted(self, vector_id: int) -> None:
        """Registers a vector ID as soft-deleted."""
        self.deleted_ids.add(vector_id)

    def is_deleted(self, vector_id: int) -> bool:
        """Checks if a vector ID has been marked as deleted."""
        return vector_id in self.deleted_ids

    def should_compact(self, total_vectors: int) -> bool:
        """Determines whether compaction should trigger based on the deletion ratio."""
        if total_vectors == 0:
            return False
        deletion_ratio = len(self.deleted_ids) / total_vectors
        return deletion_ratio >= self.compaction_threshold

    def compact_index(self, index: Any) -> int:
        """
        Purges soft-deleted vectors from an index, rebuilding contiguous arrays
        and re-indexing active vector offset mappings.
        Returns the count of purged vectors.
        """
        # Separate type guards to satisfy static type analyzers (Pylance/Pyright)
        if index.vectors is None:
            return 0
        if len(index.vectors) == 0:
            return 0

        active_mask = ~index.tombstones
        num_purged = int(np.sum(index.tombstones))

        if num_purged == 0:
            return 0

        # 1. Filter vectors array down to active non-deleted elements
        new_vectors = index.vectors[active_mask].copy()

        # 2. Rebuild ID and offset mappings
        new_id_to_offset: Dict[int, int] = {}
        new_offset_to_id: Dict[int, int] = {}
        new_offset = 0

        for old_offset, is_deleted in enumerate(index.tombstones):
            if not is_deleted:
                vector_id = index.offset_to_id[old_offset]
                new_id_to_offset[vector_id] = new_offset
                new_offset_to_id[new_offset] = vector_id
                new_offset += 1

        # 3. Update index attributes
        index.vectors = new_vectors
        index.tombstones = np.zeros(len(new_vectors), dtype=bool)
        index.id_to_offset = new_id_to_offset
        index.offset_to_id = new_offset_to_id
        index.next_offset = new_offset
        self.deleted_ids.clear()

        # 4. If index has inverted lists (IVF-Flat), rebuild cluster inverted list mappings
        if hasattr(index, "inverted_lists") and hasattr(index, "kmeans") and getattr(index, "is_trained", False):
            n_centroids = len(index.kmeans.centroids) if hasattr(index.kmeans, "centroids") else getattr(index, "nlist", 100)
            index.inverted_lists = {c: [] for c in range(n_centroids)}

            if len(new_vectors) > 0 and hasattr(index.kmeans, "centroids"):
                assignments = np.argmax(np.dot(new_vectors, index.kmeans.centroids.T), axis=1)
                for offset, cluster_id in enumerate(assignments):
                    index.inverted_lists[cluster_id].append(offset)

        return num_purged