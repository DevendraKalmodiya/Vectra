import numpy as np
from typing import Tuple

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes row-wise L2 normalization: v = v / max(||v||_2, eps).
    Ensures dot products equal cosine similarity.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / max(norm, eps)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)

def batch_cosine_similarity(query: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates 1D array of cosine similarities between a normalized 1D query vector (D,)
    and a normalized 2D target matrix (N, D) using vectorized matrix-vector inner products.
    """
    return np.dot(target_matrix, query)

def batch_matrix_similarity(queries: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    """
    Executes vectorized matrix multiplication for batch query evaluation (Q x D) @ (D x N) -> (Q x N).
    """
    return np.dot(queries, target_matrix.T)

def top_k_selection(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts top-K indices and scores using np.argpartition for O(N + K log K) performance.
    """
    n = len(scores)
    if k >= n:
        sorted_indices = np.argsort(-scores)
        return sorted_indices, scores[sorted_indices]
    
    # Unsorted partition of top-K values
    partitioned_idx = np.argpartition(-scores, k)[:k]
    # Sort only the top-K subset
    top_k_sorted = partitioned_idx[np.argsort(-scores[partitioned_idx])]
    return top_k_sorted, scores[top_k_sorted]