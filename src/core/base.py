from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SearchResult:
    ids: List[int]
    scores: List[float]
    latency_ms: float
    candidates_searched: int
    distance_calcs: int
    selected_clusters: List[int] = field(default_factory=list)
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class IndexStats:
    total_vectors: int
    dimension: int
    deleted_vectors: int
    index_type: str
    is_trained: bool


class BaseVectorIndex(ABC):
    """Abstract interface enforcing identical CRUD contracts for Exact and ANN indices."""

    def __init__(self, dimension: int):
        self.dimension = dimension

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Initialize or train the index using an N x D float32 matrix."""
        pass

    @abstractmethod
    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        """Insert a single vector into the index."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 10, **kwargs: Any) -> SearchResult:
        """Search for top_k nearest neighbors given a query vector."""
        pass

    @abstractmethod
    def delete(self, vector_id: int) -> bool:
        """Soft-delete a vector ID using a tombstone mechanism."""
        pass

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Return runtime diagnostic metrics."""
        pass