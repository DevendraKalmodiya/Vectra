import numpy as np
import pytest
from src.core.distance import l2_normalize, batch_cosine_similarity, batch_matrix_similarity, top_k_selection

def test_l2_normalize_vector_and_matrix():
    # Test 1D vector normalization
    vec = np.array([3.0, 4.0], dtype=np.float32)
    norm_vec = l2_normalize(vec)
    assert np.isclose(np.linalg.norm(norm_vec), 1.0, atol=1e-5)

    # Test 2D matrix normalization
    mat = np.random.randn(50, 384).astype(np.float32)
    norm_mat = l2_normalize(mat)
    row_norms = np.linalg.norm(norm_mat, axis=1)
    np.testing.assert_allclose(row_norms, 1.0, rtol=1e-5)

def test_cosine_similarity_correctness():
    v1 = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v2 = l2_normalize(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v3 = l2_normalize(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    
    matrix = np.vstack([v1, v2, v3])
    scores = batch_cosine_similarity(v1, matrix)
    
    np.testing.assert_allclose(scores, np.array([1.0, 0.0, 1.0]), atol=1e-5)

def test_batch_matrix_similarity():
    queries = l2_normalize(np.random.randn(5, 128).astype(np.float32))
    targets = l2_normalize(np.random.randn(100, 128).astype(np.float32))
    
    res = batch_matrix_similarity(queries, targets)
    assert res.shape == (5, 100)

def test_top_k_selection_order():
    scores = np.array([0.15, 0.88, 0.42, 0.99, 0.01, 0.73], dtype=np.float32)
    indices, top_scores = top_k_selection(scores, k=3)
    
    np.testing.assert_array_equal(indices, np.array([3, 1, 5]))
    np.testing.assert_allclose(top_scores, np.array([0.99, 0.88, 0.73]), rtol=1e-5)