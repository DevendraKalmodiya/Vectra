import numpy as np
import pytest
from src.core.ivf_flat import IVFFlatIndex

def test_ivf_flat_build_and_search():
    np.random.seed(42)
    vectors = np.random.randn(1000, 384).astype(np.float32)
    
    ivf = IVFFlatIndex(dimension=384, nlist=20, nprobe=5, seed=42)
    ivf.build(vectors)

    query = vectors[10]
    result = ivf.search(query, top_k=5)

    assert len(result.ids) == 5
    assert result.ids[0] == 10  # Exact match vector should rank first
    assert np.isclose(result.scores[0], 1.0, atol=1e-5)
    # Candidate count in IVF search should be significantly less than full matrix size (1000)
    assert result.candidates_searched < 1000

def test_ivf_nprobe_recall_effect():
    np.random.seed(42)
    vectors = np.random.randn(2000, 384).astype(np.float32)
    
    ivf = IVFFlatIndex(dimension=384, nlist=50, nprobe=1, seed=42)
    ivf.build(vectors)

    query = vectors[50]
    res_probe1 = ivf.search(query, top_k=10, nprobe=1)
    res_probe10 = ivf.search(query, top_k=10, nprobe=10)

    # Higher nprobe searches more candidates
    assert res_probe10.candidates_searched > res_probe1.candidates_searched

def test_ivf_insert_and_delete():
    np.random.seed(42)
    vectors = np.random.randn(200, 384).astype(np.float32)
    ivf = IVFFlatIndex(dimension=384, nlist=10, nprobe=3)
    ivf.build(vectors)

    new_vec = np.random.randn(384).astype(np.float32)
    new_id = 9999
    ivf.insert(new_id, new_vec)

    res = ivf.search(new_vec, top_k=1, nprobe=10)
    assert res.ids[0] == new_id

    # Soft-delete inserted vector
    deleted = ivf.delete(new_id)
    assert deleted is True

    res_after = ivf.search(new_vec, top_k=1, nprobe=10)
    assert res_after.ids[0] != new_id