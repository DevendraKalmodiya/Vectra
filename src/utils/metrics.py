import time
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ExecutionTimer:
    """Context manager and utility class to track microsecond execution latencies."""
    start_time: float = field(default=0.0)
    elapsed_ms: float = field(default=0.0)

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0


def calculate_percentiles(latencies_ms: List[float]) -> dict:
    """Computes p50, p95, and mean latency stats from a run history."""
    if not latencies_ms:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(np.mean(arr))
    }
