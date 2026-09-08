import numpy as np
from typing import Tuple
from src.core.distance import l2_normalize

class KMeansCluster:
    """
    Custom vectorized K-Means implementation using NumPy.
    Clusters L2-normalized vectors into Voronoi cells via cosine similarity.
    Includes K-Means++ initialization and empty-cluster auto-reseeding.
    """

    def __init__(self, nlist: int = 100, max_iter: int = 20, tol: float = 1e-4, seed: int = 42):
        self.nlist = nlist
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: np.ndarray = np.array([])  # Centroid matrix shape: (nlist, D)

    def _init_centroids_kmeans_plus_plus(self, vectors: np.ndarray) -> np.ndarray:
        """K-Means++ initialization for distance-weighted centroid seed placement."""
        np.random.seed(self.seed)
        n, d = vectors.shape
        centroids = np.zeros((self.nlist, d), dtype=np.float32)
        
        # Select first centroid randomly
        first_idx = np.random.choice(n)
        centroids[0] = vectors[first_idx]

        # Choose remaining centroids based on squared distance probability
        for k in range(1, self.nlist):
            sims = np.dot(vectors, centroids[:k].T)  # (N, k) cosine similarities
            dists = np.maximum(0.0, 1.0 - sims)      # Convert to cosine distance
            min_dists = np.min(dists, axis=1)
            
            probs = min_dists ** 2
            prob_sum = np.sum(probs)
            if prob_sum > 0:
                probs /= prob_sum
            else:
                probs = np.ones(n) / n

            next_idx = np.random.choice(n, p=probs)
            centroids[k] = vectors[next_idx]

        return centroids

    def fit(self, vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits K-Means centroids on N x D vectors.
        
        Returns:
            centroids: (nlist, D) array of unit-normalized cluster centroids.
            assignments: (N,) integer array mapping vector indices to centroid IDs.
        """
        n, d = vectors.shape
        nlist = min(self.nlist, n)
        self.nlist = nlist

        norm_vectors = l2_normalize(vectors.astype(np.float32))
        centroids = self._init_centroids_kmeans_plus_plus(norm_vectors)
        assignments = np.zeros(n, dtype=np.int64)

        for iteration in range(self.max_iter):
            # Calculate similarity matrix: (N, nlist)
            similarities = np.dot(norm_vectors, centroids.T)
            assignments = np.argmax(similarities, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c in range(nlist):
                members = norm_vectors[assignments == c]
                if len(members) > 0:
                    new_centroids[c] = np.mean(members, axis=0)
                else:
                    # Handle empty cluster: reseed with vector furthest from current centroids
                    max_sims = np.max(similarities, axis=1)
                    furthest_idx = np.argmin(max_sims)
                    new_centroids[c] = norm_vectors[furthest_idx]

            # Re-normalize new centroids to unit length
            new_centroids = l2_normalize(new_centroids)

            centroid_shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.centroids = centroids
        return centroids, assignments

    def predict_centroids(self, query: np.ndarray, nprobe: int = 4) -> np.ndarray:
        """Returns the top nprobe nearest cluster centroid IDs for a query vector."""
        norm_query = l2_normalize(query.astype(np.float32))
        similarities = np.dot(self.centroids, norm_query)
        top_centroids = np.argsort(-similarities)[:nprobe]
        return top_centroids