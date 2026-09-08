import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Vector & Dataset Dimensions
VECTOR_DIM = 384
TOTAL_VECTORS = 50_000
TOTAL_QUERIES = 500
DEFAULT_TOP_K = 10
RANDOM_SEED = 42

# IVF-Flat Index Hyperparameters
DEFAULT_NLIST = 100
DEFAULT_NPROBE = 4
KMEANS_MAX_ITER = 20
KMEANS_TOLERANCE = 1e-4

# Persistence Paths
VECTOR_STORE_PATH = DATA_DIR / "vectors.npy"
QUERY_STORE_PATH = DATA_DIR / "queries.npy"
TEXT_STORE_PATH = DATA_DIR / "raw_texts.json"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.npy"
