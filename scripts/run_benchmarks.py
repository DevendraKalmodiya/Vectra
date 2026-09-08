import json
import time
import numpy as np
from typing import Dict, List, Any
from src.core.exact import ExactIndex
from src.core.ivf_flat import IVFFlatIndex
from src.utils.logger import get_logger
import config

logger = get_logger("vectra.benchmarks")

def calculate_recall_at_k(predicted_ids: List[int], ground_truth_ids: np.ndarray, k: int) -> float:
    """Calculates Recall@K by comparing predicted IDs against exact ground-truth IDs."""
    gt_set = set(ground_truth_ids[:k])
    pred_set = set(predicted_ids[:k])
    if not gt_set:
        return 0.0
    return len(gt_set.intersection(pred_set)) / len(gt_set)

def run_benchmarks() -> Dict[str, Any]:
    logger.info("Loading precomputed dataset, queries, and ground truth...")
    vectors = np.load(config.VECTOR_STORE_PATH)
    queries = np.load(config.QUERY_STORE_PATH)
    ground_truth = np.load(config.GROUND_TRUTH_PATH)

    num_queries = len(queries)
    top_k = config.DEFAULT_TOP_K

    # 1. Benchmark Exact Index (Baseline Ground Truth Performance)
    logger.info("Building ExactIndex baseline...")
    exact_index = ExactIndex(dimension=config.VECTOR_DIM)
    exact_index.build(vectors)

    exact_latencies = []
    for q in queries:
        res = exact_index.search(q, top_k=top_k)
        exact_latencies.append(res.latency_ms)

    exact_results = {
        "index_type": "ExactIndex",
        "recall_at_10": 1.0,
        "latency_p50_ms": float(np.percentile(exact_latencies, 50)),
        "latency_p95_ms": float(np.percentile(exact_latencies, 95)),
        "avg_candidates_searched": int(len(vectors)),
        "speedup_vs_exact": 1.0
    }
    exact_p50 = exact_results["latency_p50_ms"]
    logger.info(f"ExactIndex Baseline -> p50: {exact_results['latency_p50_ms']:.2f} ms | p95: {exact_results['latency_p95_ms']:.2f} ms")

    # 2. Build IVF-Flat Index
    logger.info(f"Building IVFFlatIndex (nlist={config.DEFAULT_NLIST})...")
    ivf_index = IVFFlatIndex(dimension=config.VECTOR_DIM, nlist=config.DEFAULT_NLIST, seed=config.RANDOM_SEED)
    
    t0 = time.perf_counter()
    ivf_index.build(vectors)
    build_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(f"IVFFlatIndex trained and built in {build_ms:.2f} ms.")

    # 3. Sweep nprobe Values across IVF Search Space
    nprobe_candidates = [1, 2, 4, 8, 16, 32, 64, 100]
    ivf_sweep_results = []

    for nprobe in nprobe_candidates:
        recalls = []
        latencies = []
        candidates_list = []

        for q_idx in range(num_queries):
            query_vec = queries[q_idx]
            gt_ids = ground_truth[q_idx]

            res = ivf_index.search(query_vec, top_k=top_k, nprobe=nprobe)
            
            recall = calculate_recall_at_k(res.ids, gt_ids, top_k)
            recalls.append(recall)
            latencies.append(res.latency_ms)
            candidates_list.append(res.candidates_searched)

        avg_recall = float(np.mean(recalls))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        avg_candidates = float(np.mean(candidates_list))
        speedup = exact_p50 / p50_lat if p50_lat > 0 else 0.0

        metrics = {
            "nprobe": nprobe,
            "recall_at_10": round(avg_recall, 4),
            "latency_p50_ms": round(p50_lat, 3),
            "latency_p95_ms": round(p95_lat, 3),
            "avg_candidates_searched": int(avg_candidates),
            "speedup_vs_exact": round(speedup, 2)
        }
        ivf_sweep_results.append(metrics)
        logger.info(
            f"IVF nprobe={nprobe:3d} | Recall@10: {avg_recall*100:6.2f}% | "
            f"p50: {p50_lat:6.2f} ms | p95: {p95_lat:6.2f} ms | "
            f"Candidates: {int(avg_candidates):5d} | Speedup: {speedup:5.1f}x"
        )

    benchmark_payload = {
        "dataset_size": len(vectors),
        "dimension": config.VECTOR_DIM,
        "total_queries": num_queries,
        "exact_baseline": exact_results,
        "ivf_sweep": ivf_sweep_results
    }

    # 4. Save Results to Disk for Pareto Dashboard
    output_path = config.DATA_DIR / "benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    logger.info(f"Saved benchmark results to '{output_path}'.")
    logger.info("✓ Phase 7 & 8 Benchmarking complete!")
    return benchmark_payload

if __name__ == "__main__":
    run_benchmarks()