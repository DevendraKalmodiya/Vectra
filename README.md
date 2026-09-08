```markdown
# Vectra: In-Memory Vector Search Engine

Vectra is a high-performance, zero-external-dependency in-memory vector database implemented in pure Python and NumPy. It is built to systematically compare exact brute-force vector search ($O(N \cdot D)$) against an approximate nearest neighbor (ANN) Inverted File Index (IVF-Flat).

---

## 🏗️ System Architecture


```

```
                   ┌──────────────┐
                   │   Browser    │
                   │  Streamlit   │
                   └──────┬───────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   FastAPI    │
                   └──────┬───────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Search Service  │
                 └────────┬────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
     ┌──────────────┐           ┌──────────────┐
     │ ExactIndex   │           │  IVF-Flat    │
     │ (Ground      │           │  (K-Means &  │
     │  Truth)      │           │   nprobe)    │
     └──────┬───────┘           └──────┬───────┘
            │                          │
            └───────────┬──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Distance Math│
                 │  (NumPy)     │
                 └──────────────┘

```

```

---

## 🚦 Implementation Roadmap & Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Project Foundation, Environment, Config, Logger & Metrics | ✅ Completed & Verified |
| **Phase 2** | Vector Mathematics Engine (`distance.py`) & Math Tests | ✅ Completed & Verified |
| **Phase 3** | Abstract Base Class & Exact Search Engine (`exact.py`) | ✅ Completed & Verified |
| **Phase 4** | Corpus Embeddings & Dataset Generator (`prepare_data.py`) | ✅ Completed & Verified |
| **Phase 5** | Vectorized K-Means Partitioning Engine (`kmeans.py`) | ✅ Completed & Verified |
| **Phase 6** | Handcrafted IVF-Flat Inverted Index (`ivf_flat.py`) | ⏳ Pending |
| **Phase 7 & 8**| Ground Truth Recall@K & Latency Benchmark Engine | ⏳ Pending |
| **Phase 9** | Tombstone Soft-Deletion Manager | ⏳ Pending |
| **Phase 10**| Search Service Layer & FastAPI REST API | ⏳ Pending |
| **Phase 11**| Streamlit Interactive Pareto Frontier Dashboard | ⏳ Pending |
| **Phase 12**| End-to-End Testing & Demonstration Setup | ⏳ Pending |

---

## 📂 Repository Layout

```text
Vectra/
│
├── README.md
├── requirements.txt
├── config.py
│
├── data/
│   ├── raw_texts.json
│   ├── vectors.npy
│   ├── queries.npy
│   └── ground_truth.npy
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── distance.py
│   │   ├── exact.py
│   │   ├── kmeans.py
│   │   └── ivf_flat.py
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── tombstone.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   └── metrics.py
│   │
│   ├── service/
│   │   ├── __init__.py
│   │   └── search_service.py
│   │
│   └── api/
│       ├── __init__.py
│       ├── schemas.py
│       └── main.py
│
├── dashboard/
│   └── app.py
│
├── scripts/
│   ├── prepare_data.py
│   └── run_benchmarks.py
│
└── tests/
    ├── test_math.py
    ├── test_exact.py
    ├── test_kmeans.py
    ├── test_ivf.py
    └── test_api.py

```

---

## ⚡ Execution & Test Runbook

### 1. Data Pipeline Execution

```bash
python -m scripts.prepare_data

```

### 2. Unit Testing Suite

```bash
python -m pytest tests/ -v

```

```

```
