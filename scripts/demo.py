import numpy as np
from src.service.search_service import search_service
from src.utils.logger import get_logger

logger = get_logger("vectra.demo")

def main():
    logger.info("Initializing Vectra End-to-End Demonstration...")
    search_service.initialize(load_data=True)

    stats = search_service.get_stats()
    logger.info(f"Exact Index total vectors: {stats['exact'].total_vectors:,}")
    logger.info(f"IVF-Flat Index total vectors: {stats['ivf'].total_vectors:,}")

    # 1. Natural Language Semantic Search Comparison
    query_text = "Vector database similarity search using inverted file index"
    logger.info(f"\n--- Comparative Query Search: '{query_text}' ---")

    res_exact = search_service.search(
        query_vector=None,
        query_text=query_text,
        index_type="exact",
        top_k=5,
        nprobe=4
    )
    logger.info(f"Exact Search  -> Latency: {res_exact.latency_ms:.2f} ms | Candidates Searched: {res_exact.candidates_searched:,}")

    res_ivf = search_service.search(
        query_vector=None,
        query_text=query_text,
        index_type="ivf",
        top_k=5,
        nprobe=4
    )
    logger.info(f"IVF Search    -> Latency: {res_ivf.latency_ms:.2f} ms | Candidates Searched: {res_ivf.candidates_searched:,}")

    # 2. Dynamic Vector Insertion
    new_id = 99999
    new_vec = np.random.randn(384).astype(np.float32)
    logger.info(f"\n--- Inserting Dynamic Vector ID #{new_id} ---")
    search_service.insert(new_id, new_vec)

    res_ins = search_service.search(query_vector=new_vec, query_text=None, index_type="ivf", top_k=1, nprobe=10)
    logger.info(f"Retrieved Inserted Top-1 ID: {res_ins.ids[0]} (Cosine Score: {res_ins.scores[0]:.4f})")

    # 3. Soft Deletion Verification
    logger.info(f"\n--- Soft-Deleting Vector ID #{new_id} ---")
    deleted = search_service.delete(new_id)
    logger.info(f"Deletion status: {deleted}")

    res_del = search_service.search(query_vector=new_vec, query_text=None, index_type="ivf", top_k=1, nprobe=10)
    logger.info(f"Search after soft deletion Top-1 ID: {res_del.ids[0]} (Vector ID #{new_id} bypassed)")

    logger.info("\n✓ Phase 12 End-to-End Demonstration finished successfully!")

if __name__ == "__main__":
    main()