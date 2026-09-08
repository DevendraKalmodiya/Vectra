Absolutely — here’s the **updated README after Phase 3**, keeping the documentation technically accurate and aligned with the current implementation.

I also fixed the earlier wording **“zero-external-dependency”** to **“zero external vector-search dependencies”**, which is much more defensible.

````markdown
# Vectra: In-Memory Vector Search Engine

Vectra is an in-memory vector search engine built with **Python and NumPy**, designed to demonstrate the internal mechanics of exact and approximate nearest-neighbor (ANN) search.

The project implements **exact brute-force search** as the ground-truth baseline and will implement a handcrafted **IVF-Flat index** for approximate search, without relying on Pinecone, FAISS, Chroma, `sklearn.neighbors`, or other vector-search libraries.

The core objective is to measure the trade-off between **search accuracy, latency, and computational cost**.

---

## 🏗️ Project Architecture

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

Vector databases usually hide the underlying mechanics of vector search behind a simple API.

Vectra goes one level deeper by implementing the fundamental search components itself.

The system is designed around two search strategies:

```text
                 Query Vector
                      │
             ┌────────┴────────┐
             ▼                 ▼
       Exact Search        IVF-Flat
             │                 │
             ▼                 ▼
        Ground Truth      Approximate
             │              Results
             └────────┬────────┘
                      ▼
                 Recall@K
```

The exact index provides the reference answer, while IVF-Flat attempts to find similar results while examining significantly fewer vectors.

The central question is:

> **How much computation can we eliminate while preserving most of the accuracy of exact nearest-neighbor search?**

---

# 🚦 Phase Implementation Status

| Phase        | Description                                                 | Status                 |
| :----------- | :---------------------------------------------------------- | :--------------------- |
| **Phase 1**  | Project Foundation, Environment, Config, Logger & Metrics   | ✅ Completed & Verified |
| **Phase 2**  | Vector Mathematics Engine (`distance.py`) & Math Tests      | ✅ Completed & Verified |
| **Phase 3**  | Abstract Index Interface & Exact Search Engine (`exact.py`) | ✅ Completed & Verified |
| **Phase 4**  | Corpus Embeddings & Dataset Generator (`prepare_data.py`)   | ⏳ Pending              |
| **Phase 5**  | Vectorized K-Means Partitioning Engine (`kmeans.py`)        | ⏳ Pending              |
| **Phase 6**  | Handcrafted IVF-Flat Inverted Index (`ivf_flat.py`)         | ⏳ Pending              |
| **Phase 7**  | Ground Truth & Recall@K Evaluation                          | ⏳ Pending              |
| **Phase 8**  | Latency & Performance Benchmark Engine                      | ⏳ Pending              |
| **Phase 9**  | Tombstone Soft-Deletion Manager                             | ⏳ Pending              |
| **Phase 10** | Search Service Layer & FastAPI REST API                     | ⏳ Pending              |
| **Phase 11** | Streamlit Interactive Benchmark Dashboard                   | ⏳ Pending              |
| **Phase 12** | End-to-End Testing & Demonstration Setup                    | ⏳ Pending              |

---

# 📂 Project Directory Structure

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

# 🧮 Core Algorithms

## 1. Vector Mathematics

The mathematical foundation of Vectra is implemented using NumPy.

The current vector mathematics layer provides:

* L2 normalization
* Cosine similarity
* Vector dot products
* Matrix-vector operations
* Matrix multiplication
* Top-K selection
* Numerical edge-case handling

For normalized vectors, cosine similarity can be computed as:

```text
similarity(q, x) = q · x
```

These operations form the computational foundation for both exact and approximate search.

---

# 2. Exact Brute-Force Search

The `ExactIndex` compares a query vector against every stored vector.

For `N` vectors of dimension `D`, the query complexity is approximately:

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
          Top-K selection
                │
                ▼
             Results
```

Although brute-force search becomes expensive as the dataset grows, it has an important role in Vectra:

> **Exact search is the ground truth.**

Every approximate-search result can be compared against the exact result to calculate recall.

---

# 3. Index Abstraction

Vectra defines a common index interface so different search strategies can be used through the same API.

Conceptually:

```text
                VectorIndex
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     ExactIndex           IVFFlatIndex
     (Exact)              (Approximate)
```

This separation allows the search implementation to evolve without coupling the rest of the system to a specific indexing strategy.

The interface establishes the foundation for operations such as:

```text
insert()
search()
delete()
```

as additional index functionality is implemented.

---

# 🎯 IVF-Flat

IVF-Flat will provide the approximate nearest-neighbor component of Vectra.

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

During a query:

1. Find the closest cluster centroids.
2. Select the closest `nprobe` clusters.
3. Retrieve vectors from those clusters.
4. Compute exact similarity against those candidates.
5. Select the final top-K results.

The approximation comes from searching only a subset of the dataset.

---

# 🎚️ The `nprobe` Trade-off

`nprobe` controls how many IVF clusters are searched.

```text
Higher nprobe
      │
      ├── More clusters searched
      ├── More candidates
      ├── More computation
      └── Higher recall

Lower nprobe
      │
      ├── Fewer clusters searched
      ├── Fewer candidates
      ├── Lower computation
      └── Potentially lower recall
```

This creates the primary accuracy-versus-performance trade-off that Vectra will measure.

---

# 📊 Evaluation Methodology

The evaluation pipeline will use `ExactIndex` as the ground-truth reference.

```text
                    Dataset
                       │
              ┌────────┴────────┐
              ▼                 ▼
        ExactIndex          IVF-Flat
              │                 │
              ▼                 ▼
        Ground Truth       ANN Results
              │                 │
              └────────┬────────┘
                       ▼
                  Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Recall        Latency    Candidates
```

The benchmark suite will measure:

* Recall@K
* Search latency
* Index build time
* Number of candidates examined
* Number of similarity calculations
* Candidate reduction
* Memory usage
* Effect of `nprobe`

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

Three of the five ground-truth neighbors were recovered:

```text
Recall@5 = 3 / 5 = 0.60
```

This allows approximate search quality to be measured objectively.

---

# 🗑️ Soft Deletion

Vectra will support logical deletion using tombstones.

Instead of immediately restructuring an index, deleted vectors can be marked as inactive:

```text
                Vector ID
                    │
                    ▼
             Tombstone Mask
                /       \
               /         \
              ▼           ▼
           Active       Deleted
```

This avoids expensive immediate index reconstruction.

Physical cleanup and index compaction are planned as future extensions.

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

The pip path should point to the project's virtual environment:

```text
Vectra\venv\Lib\site-packages\pip
```

---

# 🧪 Verification

## Completed Phases: 1–3

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

The current tests cover the implemented mathematical and exact-search components.

### Phase 2

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

### Phase 3

```bash
python -m pytest tests/test_exact.py -v
```

Verifies:

* Exact nearest-neighbor correctness
* Top-K results
* Query behavior
* Index behavior
* Edge cases

The exact index will later serve as the reference implementation for validating IVF-Flat.

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

Actual performance depends on dataset size, vector dimensionality, data distribution, hardware, and index parameters.

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

NumPy is used for vectorized numerical computation.

The indexing and search algorithms themselves are implemented as part of Vectra.

---

# 🖥️ Interactive Dashboard

The planned Streamlit dashboard will provide an interactive laboratory for comparing exact and approximate search.

The interface will expose parameters such as:

* Query
* Index type
* Top-K
* `nprobe`
* Dataset configuration

and display metrics including:

```text
Search Results
Recall@K
Latency
Candidates Examined
Candidate Reduction
```

The goal is to make the ANN trade-off visible rather than presenting only final search results.

---

# 🔬 Benchmark Philosophy

Vectra will not hard-code performance claims.

Benchmark results will be generated from the actual implementation and hardware.

The final benchmark will allow comparisons such as:

```text
nprobe = 1
    ↓
Low computation
Low latency
Potentially lower recall

nprobe = 5
    ↓
More computation
Higher recall

nprobe = 10
    ↓
More computation
Higher recall

nprobe = ...
```

This makes the final performance claims reproducible and defensible during evaluation.

---

# 🌱 Future Extensions

The MVP focuses first on implementing and validating the core search algorithms.

Potential extensions include:

### Adaptive `nprobe`

Automatically select `nprobe` based on query characteristics or a target recall.

### Batch Search

Process multiple queries together using matrix operations.

### Memory-Mapped Storage

Use NumPy `memmap` to support datasets larger than available RAM.

### Write-Ahead Logging

Add durable mutation logging and crash recovery.

### Index Compaction

Periodically rebuild index structures to physically remove tombstoned vectors.

### Distributed / Sharded Search

Partition the dataset across workers and merge their top-K results.

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

Each phase is implemented and verified before moving to the next layer.

---

# 📌 Current Status

### Phase 1 — Completed ✅

Project foundation, environment setup, configuration, logging, and metrics infrastructure are implemented and verified.

### Phase 2 — Completed ✅

The vector mathematics engine is implemented and tested, providing the numerical foundation for vector search.

### Phase 3 — Completed ✅

The abstract index interface and exact brute-force search engine are implemented and verified through automated tests.

### Next — Phase 4 ⏳

Build the corpus embedding and dataset-generation pipeline required to create the benchmark workload for Vectra.

---

# 💡 The Core Idea

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

Exact search tells us the correct answer.

IVF-Flat attempts to find nearly the same answer while examining far fewer vectors.

The benchmark tells us **how much computation we save and how much accuracy we retain.**

---

## License

This project is intended for educational and engineering demonstration purposes.

```

### One important thing before committing this

Your current README says:

> `Vectra is a high-performance, zero-external-dependency...`

I'd **definitely not keep that wording**. At Phase 3, you haven't benchmarked performance yet, and the project does have external application/testing dependencies.

The opening above is safer and stronger:

> **“Vectra is an in-memory vector search engine built with Python and NumPy…”**

Then, once Phase 8 gives you actual benchmark results, we can make the README much more impressive with a **real benchmark table and recall-vs-latency chart** instead of making performance claims upfront.
```
