import numpy as np
import pytest
from src.core.exact import ExactIndex

def test_exact_index_build_and_search():
    np.random.seed(42)
    vectors = np.random.randn(1000, 384).astype(np.float32)
    index = ExactIndex(dimension=384)
    index.build(vectors)

    # Search using the 10th vector as query
    query = vectors[10]
    result = index.search(query, top_k=5)

    assert len(result.ids) == 5
    assert result.ids[0] == 10
    assert np.isclose(result.scores[0], 1.0, atol=1e-5)
    assert result.candidates_searched == 1000
    assert result.distance_calcs == 1000

def test_exact_index_insert():
    np.random.seed(42)
    vectors = np.random.randn(100, 384).astype(np.float32)
    index = ExactIndex(dimension=384)
    index.build(vectors)

    new_vector = np.random.randn(384).astype(np.float32)
    new_id = 999
    index.insert(new_id, new_vector)

    stats = index.get_stats()
    assert stats.total_vectors == 101

    # Search for newly inserted vector
    result = index.search(new_vector, top_k=1)
    assert result.ids[0] == new_id
    assert np.isclose(result.scores[0], 1.0, atol=1e-5)

def test_exact_index_tombstone_deletion():
    np.random.seed(42)
    vectors = np.random.randn(100, 384).astype(np.float32)
    index = ExactIndex(dimension=384)
    index.build(vectors)

    query = vectors[5]
    res1 = index.search(query, top_k=1)
    assert res1.ids[0] == 5

    # Soft delete ID 5
    deleted = index.delete(5)
    assert deleted is True

    # Search again; ID 5 must not be returned
    res2 = index.search(query, top_k=1)
    assert res2.ids[0] != 5
    
    stats = index.get_stats()
    assert stats.deleted_vectors == 1