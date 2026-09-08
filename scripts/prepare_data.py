import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from src.core.exact import ExactIndex
from src.utils.logger import get_logger
import config

logger = get_logger("vectra.prepare_data")

# Sample text domain sentences to construct real text corpus
BASE_TEXT_TEMPLATES = [
    "Python is a popular programming language for data science and web development.",
    "Machine learning models require clean training datasets and feature scaling.",
    "Neural networks use backpropagation to update weights and minimize loss.",
    "Vector databases enable fast similarity search over high-dimensional embeddings.",
    "Inverted File Index partitions vector space into Voronoi cells for ANN search.",
    "Cosine similarity measures the angle between two non-zero vectors in inner product space.",
    "FastAPI provides fast asynchronous REST API endpoints in modern Python applications.",
    "Streamlit allows building interactive data visualization applications quickly.",
    "NumPy offers C-optimized BLAS operations for matrix dot products.",
    "Euclidean distance measures straight-line distance in multidimensional space.",
    "Software engineering principles focus on modularity, readability, and testing.",
    "Distributed systems require consensus algorithms like Raft and Paxos.",
    "Database indexing drastically improves lookup performance for dynamic workloads.",
    "Natural language processing transforms raw text into semantic dense vectors.",
    "Quantization reduces memory footprint by compressing 32-bit floats into integers."
]

def generate_text_corpus(num_texts: int = 5000) -> list[str]:
    """Generates varied text snippets by combining base domain templates."""
    logger.info(f"Generating {num_texts} text corpus snippets...")
    np.random.seed(config.RANDOM_SEED)
    corpus = []
    for i in range(num_texts):
        template = np.random.choice(BASE_TEXT_TEMPLATES)
        corpus.append(f"Doc #{i+1}: {template} (Variant ID {np.random.randint(1000, 9999)})")
    return corpus

def main():
    logger.info("Starting Phase 4 Data Preparation Pipeline...")
    np.random.seed(config.RANDOM_SEED)

    # 1. Generate Raw Texts
    num_texts = 5000
    texts = generate_text_corpus(num_texts)
    
    with open(config.TEXT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2)
    logger.info(f"Saved {len(texts)} text snippets to '{config.TEXT_STORE_PATH}'.")

    # 2. Embed Real Corpus
    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (Dimension: 384)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    real_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    # 3. Scale Corpus to 50,000 Vectors via Clustered Synthetic Extension
    logger.info(f"Expanding real embeddings to target size N={config.TOTAL_VECTORS}...")
    repeats = config.TOTAL_VECTORS // num_texts
    remainder = config.TOTAL_VECTORS % num_texts
    
    expanded_vectors = []
    for r in range(repeats):
        # Add slight Gaussian perturbation to create non-identical clustered vectors
        noise = np.random.normal(0, 0.02, size=real_embeddings.shape).astype(np.float32)
        expanded_vectors.append(real_embeddings + noise)
        
    if remainder > 0:
        noise = np.random.normal(0, 0.02, size=(remainder, config.VECTOR_DIM)).astype(np.float32)
        expanded_vectors.append(real_embeddings[:remainder] + noise)

    full_vector_matrix = np.vstack(expanded_vectors).astype(np.float32)
    np.save(config.VECTOR_STORE_PATH, full_vector_matrix)
    logger.info(f"Saved {full_vector_matrix.shape} dataset vectors to '{config.VECTOR_STORE_PATH}'.")

    # 4. Generate 500 Query Vectors
    logger.info(f"Generating {config.TOTAL_QUERIES} query vectors...")
    query_texts = [f"Query search statement: {np.random.choice(BASE_TEXT_TEMPLATES)}" for _ in range(config.TOTAL_QUERIES)]
    query_vectors = model.encode(query_texts, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(config.QUERY_STORE_PATH, query_vectors)
    logger.info(f"Saved {query_vectors.shape} query vectors to '{config.QUERY_STORE_PATH}'.")

    # 5. Compute Exact Ground Truth Top-10
    logger.info("Building ExactIndex baseline over 50,000 vectors for ground-truth precomputation...")
    exact_index = ExactIndex(dimension=config.VECTOR_DIM)
    
    t0 = time.perf_counter()
    exact_index.build(full_vector_matrix)
    build_time = (time.perf_counter() - t0) * 1000.0
    logger.info(f"ExactIndex build completed in {build_time:.2f} ms.")

    logger.info(f"Computing exact top-10 ground truth for {config.TOTAL_QUERIES} queries...")
    ground_truth = np.zeros((config.TOTAL_QUERIES, config.DEFAULT_TOP_K), dtype=np.int64)
    
    t_gt = time.perf_counter()
    for q_idx in range(config.TOTAL_QUERIES):
        res = exact_index.search(query_vectors[q_idx], top_k=config.DEFAULT_TOP_K)
        ground_truth[q_idx] = res.ids
    gt_elapsed = (time.perf_counter() - t_gt) * 1000.0

    np.save(config.GROUND_TRUTH_PATH, ground_truth)
    logger.info(f"Saved ground truth array shape {ground_truth.shape} to '{config.GROUND_TRUTH_PATH}'.")
    logger.info(f"Ground truth generation completed in {gt_elapsed:.2f} ms ({gt_elapsed/config.TOTAL_QUERIES:.2f} ms/query).")
    logger.info("✓ Phase 4 Data Preparation complete!")

if __name__ == "__main__":
    main()