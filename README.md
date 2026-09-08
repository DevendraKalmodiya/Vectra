Here’s the **updated `README.md` after Phase 4**, keeping it accurate to the current state and avoiding claims about performance before Phase 8 produces actual benchmarks.

````markdown
# Vectra: In-Memory Vector Search Engine

Vectra is an in-memory vector search engine built with **Python and NumPy**, designed to systematically compare exact brute-force vector search against approximate nearest-neighbor (ANN) search using a handcrafted **IVF-Flat** index.

The project intentionally avoids existing vector-search implementations such as **Pinecone, FAISS, Chroma, and `sklearn.neighbors`**.

The core objective is to understand and measure the trade-off between **search accuracy, latency, and computational cost**.

---

## 🏗️ System Architecture

```text
                   ┌──────────────────┐
                   │     Browser      │
                   │    Streamlit     │
                   └────────┬─────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │     FastAPI      │
                   │   REST Gateway   │
                   └────────┬─────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │   Search Service   │
                  │ Unified Interface  │
                  └─────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       ┌──────────────┐           ┌──────────────┐
       │  ExactIndex  │           │   IVF-Flat   │
       │              │           │              │
       │ Ground Truth │           │ K-Means +    │
       │              │           │ nprobe       │
       └──────┬───────┘           └──────┬───────┘
              │                          │
              └────────────┬─────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Vector Math     │
                  │      Core        │
                  │      NumPy       │
                  └──────────────────┘
````

---

# 🎯 Project Objective

Vector databases usually expose vector insertion and search through a simple API while hiding the algorithms underneath.

Vectra implements those core mechanisms explicitly.

The system is built around two complementary search strategies:

```text
                    Query Vector
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Exact Search           IVF-Flat
              │                     │
              ▼                     ▼
         Ground Truth         Approximate
              │                  Results
              └──────────┬──────────┘
                         ▼
                    Recall@K
```

The exact index provides the reference result.

The IVF-Flat index attempts to retrieve the same nearest neighbors while searching only a subset of the dataset.

The central question is:

> **How much computation can we eliminate while preserving most of the accuracy of exact nearest-neighbor search?**

---

# 🚦 Implementation Roadmap & Status

| Phase        | Description                                               | Status                 |
| :----------- | :-------------------------------------------------------- | :--------------------- |
| **Phase 1**  | Project Foundation, Environment, Config, Logger & Metrics | ✅ Completed & Verified |
| **Phase 2**  | Vector Mathematics Engine (`distance.py`) & Math Tests    | ✅ Completed & Verified |
| **Phase 3**  | Abstract Base Class & Exact Search Engine (`exact.py`)    | ✅ Completed & Verified |
| **Phase 4**  | Corpus Embeddings & Dataset Generator (`prepare_data.py`) | ✅ Completed & Verified |
| **Phase 5**  | Vectorized K-Means Partitioning Engine (`kmeans.py`)      | ⏳ Pending              |
| **Phase 6**  | Handcrafted IVF-Flat Inverted Index (`ivf_flat.py`)       | ⏳ Pending              |
| **Phase 7**  | Ground Truth & Recall@K Evaluation                        | ⏳ Pending              |
| **Phase 8**  | Latency & Performance Benchmark Engine                    | ⏳ Pending              |
| **Phase 9**  | Tombstone Soft-Deletion Manager                           | ⏳ Pending              |
| **Phase 10** | Search Service Layer & FastAPI REST API                   | ⏳ Pending              |
| **Phase 11** | Streamlit Interactive Pareto Frontier Dashboard           | ⏳ Pending              |
| **Phase 12** | End-to-End Testing & Demonstration Setup                  | ⏳ Pending              |

---

# 📂 Repository Layout

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
    ├── test_ivf.py
    └── test_api.py
```

---

# 🧮 Core Components

## 1. Vector Mathematics Engine

The mathematical foundation of Vectra is implemented using NumPy.

The current implementation provides:

* L2 normalization
* Cosine similarity
* Vector dot products
* Matrix operations
* Top-K selection
* Numerical edge-case handling

For normalized vectors, cosine similarity becomes a dot product:

```text
similarity(q, x) = q · x
```

This mathematical layer is shared by the exact and approximate search implementations.

---

# 2. Exact Brute-Force Search

`ExactIndex` provides the exhaustive nearest-neighbor search implementation.

For every query, the index compares the query vector against all stored vectors.

For `N` vectors of dimension `D`, query complexity is approximately:

```text
O(N × D)
```

The process is:

```text
                  Query
                    │
                    ▼
          Compare with every
           stored vector
                    │
                    ▼
          Compute similarity
                    │
                    ▼
              Top-K
                    │
                    ▼
               Results
```

The exact index is intentionally computationally expensive because it serves as the **ground-truth reference** for ANN evaluation.

---

# 3. Corpus & Dataset Pipeline

Phase 4 introduces the dataset generation pipeline.

The pipeline prepares:

* **50,000 dataset vectors**
* **500 query vectors**
* **Top-10 ground-truth reference results**

The generated artifacts are stored in the `data/` directory.

```text
                    Corpus
                      │
                      ▼
               Embedding Pipeline
                      │
                      ▼
              50,000 Vectors
                      │
             ┌────────┴────────┐
             ▼                 ▼
        Query Generation   Exact Search
             │                 │
             ▼                 ▼
       500 Query Vectors   Ground Truth
                               │
                               ▼
                          Top-10 IDs
```

The ground-truth matrix provides the reference required for measuring IVF-Flat recall in later phases.

---

# 🎯 IVF-Flat

The next major indexing component is IVF-Flat.

The dataset will be partitioned into `nlist` clusters using K-Means.

```text
                         Dataset
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
            Centroid 1              Centroid 2
                │                       │
          ┌─────┴─────┐           ┌─────┴─────┐
          ▼           ▼           ▼           ▼
       Vector      Vector      Vector      Vector
```

At query time:

1. Identify the nearest cluster centroids.
2. Select the closest `nprobe` clusters.
3. Retrieve vectors from those clusters.
4. Compute exact similarity on the candidates.
5. Return the top-K results.

The approximation comes from searching only a subset of the available vectors.

---

# 🎚️ `nprobe`

`nprobe` determines how many IVF clusters are searched for a query.

```text
Higher nprobe
      │
      ├── More clusters searched
      ├── More candidates
      ├── More computation
      └── Potentially higher recall

Lower nprobe
      │
      ├── Fewer clusters searched
      ├── Fewer candidates
      ├── Less computation
      └── Potentially lower recall
```

This parameter will become one of the primary controls in the benchmark and dashboard phases.

---

# 📊 Evaluation Methodology

The exact index provides the ground-truth nearest neighbors.

IVF-Flat results will then be compared against that ground truth.

```text
                     Dataset
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
        ExactIndex             IVF-Flat
             │                     │
             ▼                     ▼
       Ground Truth          ANN Results
             │                     │
             └──────────┬──────────┘
                        ▼
                   Recall@K
```

The evaluation pipeline will measure:

* Recall@K
* Search latency
* Index build time
* Candidates examined
* Similarity calculations
* Candidate reduction
* Memory usage
* `nprobe` impact

---

# 📈 Recall@K

Recall@K measures how many of the true nearest neighbors are recovered by the approximate index.

```text
Recall@K =
|Approximate Results ∩ Exact Results|
-------------------------------------
                  K
```

Example:

```text
Exact: [1, 2, 3, 4, 5]

IVF:   [1, 2, 3, 7, 9]
```

Three of the five exact neighbors were recovered:

```text
Recall@5 = 3 / 5 = 0.60
```

---

# ⚡ Quickstart

## 1. Create the Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

---

## 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Verify the environment:

```bash
python --version
python -m pip --version
```

The pip path should point to the project's virtual environment.

---

# 🔬 Data Pipeline

Generate the Phase 4 dataset:

```bash
python -m scripts.prepare_data
```

This creates the cached vector and query artifacts required for subsequent phases.

Expected dataset:

```text
Dataset vectors:      50,000
Query vectors:           500
Ground truth:            500 × 10
```

---

# 🧪 Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

### Phase 2 — Vector Mathematics

```bash
python -m pytest tests/test_math.py -v
```

Verifies:

* L2 normalization
* Cosine similarity
* Dot products
* Matrix operations
* Top-K selection
* Numerical correctness

### Phase 3 — Exact Search

```bash
python -m pytest tests/test_exact.py -v
```

Verifies:

* Exact nearest-neighbor correctness
* Top-K results
* Query behavior
* Index behavior
* Edge cases

### Phase 4 — Dataset Pipeline

The dataset pipeline can be verified by executing:

```bash
python -m scripts.prepare_data
```

The generated artifacts should contain:

```text
50,000 dataset vectors
500 query vectors
500 × 10 ground-truth matrix
```

---

# 🧠 Complexity

| Operation |  Exact Index |       IVF-Flat       |
| :-------- | :----------: | :------------------: |
| Build     |   O(N × D)   |     O(I × N × D)     |
| Query     |   O(N × D)   | O(nlist × D + C × D) |
| Delete    | O(1) logical |     O(1) logical     |
| Accuracy  |     Exact    |      Approximate     |

Where:

* `N` = number of vectors
* `D` = vector dimension
* `I` = number of K-Means iterations
* `nlist` = number of IVF clusters
* `nprobe` = number of clusters searched
* `C` = number of candidate vectors examined

Actual performance will depend on dataset distribution, vector dimensionality, hardware, and index parameters.

---

# 🧱 Design Constraints

Vectra intentionally avoids existing vector-search implementations.

### Not Used

```text
Pinecone
FAISS
Chroma
sklearn.neighbors
```

### Core Technologies

```text
Python
NumPy
```

### Application & Testing

```text
FastAPI
Streamlit
pytest
```

NumPy is used for vectorized numerical operations.

The indexing algorithms and search logic are implemented within Vectra itself.

---

# 🖥️ Interactive Dashboard

The planned Streamlit dashboard will provide an interactive laboratory for comparing exact and approximate search.

The dashboard will allow users to explore parameters such as:

* Query
* Index type
* Top-K
* `nprobe`

and observe:

* Search results
* Recall@K
* Latency
* Candidates examined
* Candidate reduction

The goal is to make the ANN accuracy/performance trade-off directly observable.

---

# 🔬 Benchmark Philosophy

Vectra does not hard-code performance claims.

All latency and recall results will be generated from the actual implementation and execution environment.

The benchmark will make it possible to answer questions such as:

```text
How many vectors did exact search examine?

How many did IVF-Flat examine?

How much computation was saved?

How much recall was retained?

How does changing nprobe affect the trade-off?
```

---

# 🌱 Future Extensions

The current MVP focuses on building and validating the fundamental search engine.

Potential extensions include:

### Adaptive `nprobe`

Automatically select an appropriate `nprobe` based on query characteristics or a target recall.

### Batch Search

Process multiple queries together using matrix operations.

### Memory-Mapped Storage

Use NumPy `memmap` to support datasets larger than available RAM.

### Write-Ahead Logging

Add durable mutation logging and crash recovery.

### Index Compaction

Physically remove tombstoned vectors through periodic index rebuilding.

### Distributed / Sharded Search

Partition the index across workers and merge top-K results.

---

# 🛠️ Development Roadmap

```text
Phase 1
Project Foundation
        │
        ▼
Phase 2
Vector Mathematics
        │
        ▼
Phase 3
Exact Search
        │
        ▼
Phase 4
Dataset & Embeddings
        │
        ▼
Phase 5
K-Means
        │
        ▼
Phase 6
IVF-Flat
        │
        ▼
Phase 7–8
Recall & Performance
        │
        ▼
Phase 9
Deletion
        │
        ▼
Phase 10
API
        │
        ▼
Phase 11
Dashboard
        │
        ▼
Phase 12
Testing & Demo
```

Each phase is implemented and verified before moving to the next.

---

# 📌 Current Status

### Phase 1 — Completed ✅

Project foundation, virtual environment, configuration, logging, and metrics infrastructure are implemented and verified.

### Phase 2 — Completed ✅

The vector mathematics engine is implemented and tested, providing the numerical foundation for vector search.

### Phase 3 — Completed ✅

The abstract index interface and exact brute-force search engine are implemented and verified through automated tests.

### Phase 4 — Completed ✅

The corpus embedding and dataset-generation pipeline has been implemented and verified.

The current benchmark dataset contains:

```text
50,000 dataset vectors
500 query vectors
500 × 10 ground-truth matrix
```

These cached artifacts provide the workload required for K-Means and IVF-Flat development.

### Next — Phase 5 ⏳

Implement the vectorized K-Means partitioning engine in:

```text
src/core/kmeans.py
```

This will provide the clustering layer required by IVF-Flat.

---

# 💡 Core Insight

Vectra is ultimately about one measurable systems trade-off:

```text
                         ACCURACY
                            ▲
                            │
                     Higher nprobe
                            │
                            │
                            │
     LOWER COMPUTATION ◄────┼────► HIGHER COMPUTATION
                            │
                            │
                      Lower nprobe
                            │
                            ▼
                       LOWER RECALL
```

Exact search provides the correct answer.

IVF-Flat attempts to approximate that answer while examining fewer vectors.

The benchmark determines **how much computation can be saved while retaining high-quality search results**.

---

## License

This project is intended for educational and engineering demonstration purposes.

````

### Phase 4 Git commits

For the implementation:

```bash
git commit -m "feat: add corpus embedding and 50k vector dataset pipeline"
````

For this README update:

```bash
git commit -m "docs: update README for phase 4 completion"
```

One thing to watch: **don't commit the generated `.npy` files if your `.gitignore` currently excludes them.** The reproducible `prepare_data.py` pipeline is the important artifact; the 50k-vector cache can be regenerated.
