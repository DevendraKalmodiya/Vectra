import numpy as np
import pytest
from src.core.kmeans import KMeansCluster

def test_kmeans_fitting_and_centroids():
    np.random.seed(42)
    vectors = np.random.randn(500, 384).astype(np.float32)
    
    kmeans = KMeansCluster(nlist=10, max_iter=15, seed=42)
    centroids, assignments = kmeans.fit(vectors)

    assert centroids.shape == (10, 384)
    assert len(assignments) == 500
    assert len(np.unique(assignments)) > 1  # Verify cluster dispersion

    # Ensure all centroids are unit length
    norms = np.linalg.norm(centroids, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

def test_kmeans_predict_centroids():
    np.random.seed(42)
    vectors = np.random.randn(200, 384).astype(np.float32)
    kmeans = KMeansCluster(nlist=5, seed=42)
    kmeans.fit(vectors)

    query = vectors[0]
    top_centroids = kmeans.predict_centroids(query, nprobe=2)

    assert len(top_centroids) == 2
    assert 0 <= top_centroids[0] < 5