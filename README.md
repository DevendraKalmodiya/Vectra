# Vectra

### In-Memory Vector Search Engine Built From Scratch

Vectra is a lightweight, in-memory vector search engine implemented using **Python and NumPy**, designed to demonstrate how modern approximate nearest-neighbor (ANN) vector search works internally.

The project implements an **exact brute-force vector index** as the ground-truth baseline and an **IVF-Flat approximate index** built from scratch—without using Pinecone, FAISS, Chroma, `sklearn.neighbors`, or any other vector-search library.

The primary goal is to understand and measure the trade-off between **search accuracy and computational cost**.

---

## Why Vectra?

Vector databases often hide the underlying mechanics of vector search behind a simple API.

**Vectra removes that abstraction.**

Instead of importing a vector database, this project implements the core search pipeline directly:

```text
Text
 │
 ▼
Embedding Vector
 │
 ▼
┌───────────────────────────┐
│       Vector Index        │
├─────────────┬─────────────┤
│ Exact Search│  IVF-Flat   │
└─────────────┴─────────────┘
 │
 ▼
Top-K Nearest Neighbors
```

This makes it possible to inspect and benchmark what happens internally when an approximate index reduces the number of vectors that need to be compared.

---

## Architecture & Data Flow

```text
┌──────────────────────────────┐
│     Streamlit Laboratory     │
│       Interactive UI         │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       FastAPI Gateway        │
│        REST Interface        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       Search Service         │
│  Unified Search Abstraction  │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
┌──────────────┐ ┌──────────────┐
│ Exact Index  │ │ IVF-Flat     │
│              │ │ Index        │
└──────┬───────┘ └──────┬───────┘
       │                │
       └────────┬───────┘
                ▼
┌──────────────────────────────┐
│       Vector Math Core       │
│            NumPy             │
│ Normalization / Dot Product  │
│ Batch Cosine / Top-K         │
└──────────────────────────────┘
```

---

## Component Matrix

### Implemented — MVP

* **Vector Mathematics:** L2 normalization, cosine similarity, vectorized dot products, and top-K selection.
* **Exact Search:** Brute-force `O(N × D)` nearest-neighbor search used as ground truth.
* **ANN Search:** IVF-Flat index with K-Means clustering.
* **Storage & Lifecycle:** O(1) logical deletion using tombstone bitmasks.
* **Evaluation:** Ground-truth generation, Recall@K, latency, candidate count, and distance-computation benchmarks.
* **Service Layer:** FastAPI REST API.
* **Interactive UI:** Streamlit-based search and benchmarking laboratory.

### Future Architecture

* Adaptive `nprobe` auto-tuning
* Batch query execution
* Disk-backed `np.memmap` storage
* Binary Write-Ahead Logging (WAL)
* Crash recovery
* Index compaction
* Distributed/sharded search

---

## Core Algorithms

### Exact Brute-Force Search

The exact index compares a query vector against every vector in the dataset.

For normalized vectors, cosine similarity becomes a dot product:

```text
similarity(q, x) = q · x
```

For `N` vectors with dimension `D`, a query requires approximately:

```text
O(N × D)
```

operations.

This index provides the **ground truth** against which the approximate index is evaluated.

---

### IVF-Flat

Vectra implements an **Inverted File Index (IVF-Flat)** from scratch.

The dataset is partitioned into `nlist` clusters using K-Means.

```text
                    Dataset
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
         Centroid 1          Centroid 2
             │                   │
        ┌────┴────┐         ┌────┴────┐
        ▼         ▼         ▼         ▼
     Vector     Vector    Vector     Vector
```

At query time:

1. Find the nearest cluster centroids.
2. Select the `nprobe` closest clusters.
3. Gather vectors from those clusters.
4. Compute exact cosine similarity on those candidates.
5. Return the top-K results.

The main trade-off is controlled by `nprobe`:

```text
Higher nprobe
     │
     ├── More candidates
     ├── More computation
     └── Higher recall

Lower nprobe
     │
     ├── Fewer candidates
     ├── Lower latency
     └── Potentially lower recall
```

---

## Evaluation

Vectra uses the exact index to generate ground-truth nearest neighbors.

The IVF-Flat index is then evaluated using:

* **Recall@K**
* **Search latency**
* **Index build time**
* **Distance calculations**
* **Candidates examined**
* **Candidate reduction**
* **Memory usage**

The central question is:

> **How much computation can we eliminate while still finding almost all of the true nearest neighbors?**

---

## Recall@K

Recall@K measures the overlap between the approximate results and the exact ground-truth results.

```text
Recall@K =
|Approximate Results ∩ Exact Results|
-------------------------------------
                 K
```

For example:

```text
Exact: [1, 2, 3, 4, 5]
IVF:   [1, 2, 3, 7, 9]

Recall@5 = 3 / 5 = 0.60
```

This allows Vectra to measure ANN quality objectively rather than evaluating performance using latency alone.

---

## Soft Deletion

Vectra supports logical deletion using tombstones.

Instead of immediately restructuring the index, deleted vectors are marked as inactive:

```text
Tombstone
   │
   ├── 0 → Active
   └── 1 → Deleted
```

This allows constant-time logical deletion while avoiding immediate index reconstruction.

Physical cleanup can later be implemented through index compaction.

---

## Project Structure

```text
vectra/
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
    ├── test_ivf.py
    └── test_api.py
```

---

## Installation

```bash
git clone <repository-url>
cd vectra

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Prepare Data

Generate the vectors, queries, and ground-truth data:

```bash
python scripts/prepare_data.py
```

---

## Run Benchmarks

```bash
python scripts/run_benchmarks.py
```

The benchmark compares exact search against IVF-Flat across different `nprobe` configurations.

**Benchmark numbers are generated locally and are not hard-coded into the documentation.**

---

## Run the API

```bash
uvicorn src.api.main:app --reload
```

---

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The Streamlit interface acts as an interactive laboratory for exploring vector search.

Users can compare exact and approximate search while observing:

* Search results
* Recall@K
* Latency
* Candidates examined
* Candidate reduction
* `nprobe` effects

---

## Design Constraints

Vectra intentionally avoids existing vector-search implementations.

### Not Used

```text
Pinecone
FAISS
Chroma
sklearn.neighbors
```

### Used

```text
Python
NumPy
FastAPI
Streamlit
```

NumPy is used for vectorized numerical operations. The indexing, clustering, search, and evaluation logic are implemented by Vectra itself.

---

## Complexity

| Operation |  Exact Index |             IVF-Flat |
| --------- | -----------: | -------------------: |
| Build     |     O(N × D) |         O(I × N × D) |
| Query     |     O(N × D) | O(nlist × D + C × D) |
| Insert    |        O(1)* |    O(D + assignment) |
| Delete    | O(1) logical |         O(1) logical |
| Accuracy  |        Exact |          Approximate |

Where:

* `N` = number of vectors
* `D` = vector dimension
* `I` = K-Means iterations
* `C` = number of IVF candidates examined
* `nlist` = number of clusters
* `nprobe` = number of clusters searched per query

`*` Exact insertion complexity depends on the underlying NumPy storage strategy.

---

## Why IVF-Flat?

IVF-Flat provides a useful balance between implementation complexity and measurable ANN behavior.

Compared with HNSW, it offers:

* Simpler implementation
* Clear `nlist` / `nprobe` controls
* Exact similarity within selected clusters
* Straightforward candidate accounting
* Easy recall-vs-latency experimentation

The goal is to understand the mechanics of ANN search rather than hide them behind a production library.

---

## Future Extensions

### Adaptive `nprobe`

Automatically select `nprobe` based on the query or target recall.

### Batch Search

Support multiple queries simultaneously using matrix operations.

### Memory-Mapped Storage

Use `np.memmap` to work with datasets larger than available RAM.

### Write-Ahead Logging

Add durable mutation logging and crash recovery.

### Compaction

Periodically rebuild the index to physically remove tombstoned vectors.

### Distributed Search

Partition the index across multiple workers and merge their top-K results.

---

## Testing

Run:

```bash
pytest
```

Tests cover:

* Vector normalization
* Cosine similarity
* Exact search correctness
* IVF search behavior
* Recall calculation
* Tombstone deletion
* API behavior

The exact index acts as the reference implementation for validating approximate search.

---

## The Core Idea

Vectra is built around one simple systems question:

> **Can we significantly reduce the amount of computation required for nearest-neighbor search while preserving most of the accuracy of an exact search?**

The answer is demonstrated experimentally through exact search, IVF-Flat, configurable `nprobe`, and quantitative benchmarking.

```text
              ACCURACY
                 ▲
                 │
          Higher nprobe
                 │
                 │
                 │
LOW LATENCY ◄────┼────► HIGH COMPUTATION
                 │
                 │
           Lower nprobe
                 │
                 ▼
             LESS RECALL
```

---

## License

This project is intended for educational and engineering demonstration purposes.
