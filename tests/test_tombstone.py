import numpy as np
import pytest
from src.core.exact import ExactIndex
from src.core.ivf_flat import IVFFlatIndex
from src.storage.tombstone import TombstoneManager


def test_tombstone_manager_threshold():
    """Verify tombstone deletion ratio calculations and compaction trigger thresholds."""
    tm = TombstoneManager(compaction_threshold=0.2)
    assert not tm.should_compact(100)

    for i in range(20):
        tm.mark_deleted(i)

    assert tm.should_compact(100)


def test_exact_index_compaction():
    """Verify memory compaction and soft-deletion purging on ExactIndex."""
    np.random.seed(42)
    vectors = np.random.randn(100, 384).astype(np.float32)
    index = ExactIndex(dimension=384)
    index.build(vectors)

    tm = TombstoneManager(compaction_threshold=0.2)

    # Soft-delete 25 vectors
    for vid in range(25):
        index.delete(vid)
        tm.mark_deleted(vid)

    assert tm.should_compact(100)

    purged_count = tm.compact_index(index)
    assert purged_count == 25

    # Explicit null check narrows index.vectors type from Optional[np.ndarray] -> np.ndarray
    assert index.vectors is not None
    assert len(index.vectors) == 75
    assert len(index.tombstones) == 75
    assert index.get_stats().deleted_vectors == 0


def test_ivf_index_compaction():
    """Verify memory compaction and inverted list re-indexing on IVFFlatIndex."""
    np.random.seed(42)
    vectors = np.random.randn(100, 384).astype(np.float32)
    index = IVFFlatIndex(dimension=384, nlist=5, nprobe=2, seed=42)
    index.build(vectors)

    tm = TombstoneManager(compaction_threshold=0.2)

    for vid in range(30):
        index.delete(vid)
        tm.mark_deleted(vid)

    purged_count = tm.compact_index(index)
    assert purged_count == 30

    # Explicit null check narrows index.vectors type from Optional[np.ndarray] -> np.ndarray
    assert index.vectors is not None
    assert len(index.vectors) == 70
    assert index.get_stats().deleted_vectors == 0