# ⚡ Vectra

### An in-memory vector search engine built from scratch.

Vectra is a learning-focused but fully functional vector search engine that exposes the mechanics behind approximate nearest-neighbor search instead of hiding them behind a vector database library.

It implements **exact brute-force cosine similarity** as the ground-truth baseline and a **handcrafted IVF-Flat index** built with K-Means clustering. The system measures the central trade-off in approximate vector search:

> **How much search computation can be eliminated while preserving nearest-neighbor accuracy?**

Vectra includes a complete evaluation and visualization layer with live semantic search, Recall@K measurement, latency analysis, candidate reduction, Pareto-frontier benchmarking, IVF cluster inspection, CRUD operations, tombstone deletion, compaction, query diagnostics, and an interactive Streamlit control plane.

---

## 🚀 Installation

### Prerequisites

- Python **3.12+**
- Git
- At least **8 GB RAM** recommended for the full 50,000-vector workload
- Internet access on the first run to download the embedding model

### 1. Clone the repository

```bash
git clone https://github.com/DevendraKalmodiya/Vectra.git
cd Vectra
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Generate the dataset

Vectra uses a 50,000-vector corpus with 500 benchmark queries.

```bash
python -m scripts.prepare_data
```

This prepares the vector corpus, query vectors, and exact ground-truth results used by the benchmark pipeline.

### 5. Run the complete test suite

```bash
python -m pytest tests/ -v
```

### 6. Run the end-to-end demonstration

```bash
python -m scripts.demo
```

### 7. Start the FastAPI service

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI also provides interactive API documentation at:

```text
http://localhost:8000/docs
```

### 8. Launch the Vectra dashboard

```bash
python -m streamlit run dashboard/app.py
```

If Streamlit's file watcher encounters the known deep-import inspection issue on the local environment, launch with:

```bash
python -m streamlit run dashboard/app.py --server.fileWatcherType none
```

---

# 📌 What Is Vectra?

Modern AI applications frequently depend on vector search.

A text query is converted into an embedding:

```text
"machine learning algorithms"
            │
            ▼
      Embedding Model
            │
            ▼
       Query Vector
            │
            ▼
      Vector Search
            │
            ▼
         Top-K
```

The simplest implementation is exact brute-force search:

```text
Query
  │
  ├── Compare with Vector 1
  ├── Compare with Vector 2
  ├── Compare with Vector 3
  ├── ...
  └── Compare with Vector 50,000
              │
              ▼
           Top-K
```

It is simple and exact, but every query requires comparison against the entire active dataset.

Vectra implements a second approach:

```text
Query
  │
  ▼
Find nearest K-Means centroids
  │
  ▼
Select nprobe clusters
  │
  ▼
Retrieve candidate vectors
  │
  ▼
Compute exact cosine similarity
  │
  ▼
Top-K
```

The second approach is **IVF-Flat**.

Instead of searching the entire vector space, IVF-Flat narrows the search to selected partitions.

The resulting trade-off is measurable through:

- Recall@K
- Search latency
- Candidate count
- Candidate reduction
- P50/P95 latency
- nprobe
- Distance computations

---

# 🎯 Project Objective

Vectra is built around one question:

> **What does approximate vector search actually cost?**

The system establishes an exact brute-force implementation as ground truth and compares it against a handcrafted IVF-Flat implementation.

For every benchmark configuration, Vectra measures:

```text
                    Exact Search
                         │
                         ▼
                    Ground Truth
                         │
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Exact Search           IVF-Flat Search
              │                     │
              │                 nprobe
              │                     │
              │              Candidate Selection
              │                     │
              │                     ▼
              │              Exact Similarity
              │                     │
              └──────────┬──────────┘
                         ▼
                   Compare Results
                         │
                         ▼
                     Recall@K
```

---

# ✨ Features

## Search Engine

- Exact brute-force cosine similarity
- Handcrafted IVF-Flat ANN index
- Vectorized NumPy distance calculations
- K-Means clustering implemented from scratch
- Top-K nearest-neighbor retrieval
- Natural-language semantic search
- Configurable `nprobe`
- Configurable `Top-K`

## Evaluation

- Exact-search ground truth
- Recall@K
- P50 latency
- P95 latency
- Candidate count
- Candidate reduction
- Speedup comparison
- nprobe sweep
- Recall-vs-latency Pareto frontier
- 500-query benchmark workload

## Index Lifecycle

- Vector insertion
- Search
- Soft deletion
- Tombstone tracking
- Index compaction
- Active/deleted vector accounting

## Interactive Control Plane

The Streamlit dashboard provides:

- Live semantic search
- Exact vs IVF-Flat result comparison
- Dynamic current-query metrics
- Current-query Pareto marker
- Search-space reduction visualization
- IVF cluster inspection
- Cluster distribution analysis
- Selected-cluster diagnostics
- Query execution-stage timings
- Result-level retrieval explanation
- CRUD operations
- Tombstone and compaction visibility
- Query history
- Benchmark visualization
- Index statistics
- Architecture and algorithm explanation

---

# 🏗️ Architecture

```text
                           ┌──────────────────────┐
                           │       Browser        │
                           │      Streamlit       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │       FastAPI        │
                           │      REST API        │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   Search Service     │
                           └──────────┬───────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         ▼                         ▼
                ┌─────────────────┐       ┌─────────────────┐
                │   ExactIndex    │       │    IVF-Flat     │
                │                 │       │                 │
                │ Ground Truth   │       │ K-Means +       │
                │ Brute Force    │       │ Inverted Lists  │
                └────────┬────────┘       └────────┬────────┘
                         │                         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Distance Math     │
                           │      NumPy            │
                           └──────────────────────┘
```

Supporting components:

```text
Dataset / Embedding Pipeline
            │
            ▼
       Vector Corpus
            │
            ├──────────────► Exact Index
            │
            └──────────────► K-Means
                                  │
                                  ▼
                              IVF-Flat
                                  │
                                  ▼
                         Tombstone Manager
                                  │
                                  ▼
                             Compaction
```

---

# 🔬 How Vectra Searches

## 1. Exact Search

The exact index compares the query against every active vector.

For normalized vectors, cosine similarity can be computed as:

\[
\text{cosine}(q,x)=q\cdot x
\]

The search performs:

```text
Query
  │
  ▼
Compare against all N vectors
  │
  ▼
Compute cosine similarities
  │
  ▼
Select Top-K
```

Approximate query complexity:

\[
O(ND)
\]

where:

- \(N\) = number of vectors
- \(D\) = vector dimension

For Vectra's default workload:

```text
N = 50,000
D = 384
```

Exact search therefore provides the reference result set used to evaluate ANN recall.

---

# 🧩 2. K-Means Partitioning

Before IVF-Flat can search efficiently, the vector space is partitioned into clusters.

Vectra implements K-Means using NumPy operations.

The algorithm repeatedly performs:

```text
Initialize centroids
       │
       ▼
Assign vectors to nearest centroid
       │
       ▼
Recompute centroid positions
       │
       ▼
Check convergence
       │
       └──────► Repeat
```

The resulting clusters form the inverted lists used by IVF-Flat.

---

# ⚡ 3. IVF-Flat

IVF stands for **Inverted File**.

The dataset is divided into `nlist` clusters.

At query time:

```text
Query Vector
     │
     ▼
Compare against centroids
     │
     ▼
Select nearest nprobe centroids
     │
     ▼
Retrieve vectors from those clusters
     │
     ▼
Compute exact cosine similarity
     │
     ▼
Select Top-K
```

The key parameter is:

### `nprobe`

`nprobe` controls how many clusters are searched.

A smaller `nprobe`:

```text
Fewer clusters
      ↓
Fewer candidates
      ↓
Less distance computation
      ↓
Potentially lower recall
```

A larger `nprobe`:

```text
More clusters
      ↓
More candidates
      ↓
More computation
      ↓
Potentially higher recall
```

This makes `nprobe` the central accuracy/performance control in Vectra.

---

# 📐 Why "Flat"?

IVF-Flat does not approximate the distance calculation within the selected clusters.

Once candidate vectors have been selected, Vectra computes their cosine similarity exactly.

The approximation comes from:

> **Which vectors are considered**

not from:

> **How similarity is calculated for those vectors**

This distinction is fundamental to understanding IVF-Flat.

---

# 📊 Evaluation Methodology

Vectra uses exact brute-force search as ground truth.

The benchmark workflow is:

```text
500 Query Vectors
       │
       ├──────────────► Exact Search
       │                    │
       │                    ▼
       │              Ground Truth
       │
       └──────────────► IVF-Flat
                            │
                     nprobe sweep
                            │
                            ▼
                      ANN Results
                            │
                            ▼
                     Compare with
                     Ground Truth
                            │
                            ▼
                        Recall@10
```

## Recall@K

Recall@K measures how many of the true exact top-K neighbors were recovered by the approximate search.

\[
Recall@K =
\frac{|Exact_K \cap Approximate_K|}
{|Exact_K|}
\]

For example:

```text
Exact Top-10
A B C D E F G H I J

IVF Top-10
A B C D E F G H X Y
```

The intersection contains 8 results.

Therefore:

\[
Recall@10 = \frac{8}{10}=80\%
\]

This allows Vectra to quantify the accuracy cost of reducing the search space.

---

# 📈 Pareto Frontier

Vectra evaluates multiple `nprobe` configurations.

A typical benchmark explores:

```text
nprobe = 1
nprobe = 2
nprobe = 4
nprobe = 8
nprobe = 16
nprobe = 32
...
```

Each configuration produces:

```text
nprobe
   │
   ├── Recall@10
   ├── P50 latency
   ├── P95 latency
   └── Candidate count
```

The dashboard visualizes these results as:

```text
Recall@10
   │
1.0│                         ●
   │                    ●
0.9│               ●
   │
0.8│          ●
   │
0.6│     ●
   │
   └──────────────────────────────►
              Latency
```

The dashboard also places the **current live query** directly onto the global benchmark curve.

This separates:

- **Global benchmark behavior**
- **Current query behavior**

without rerunning the complete benchmark after every search.

---

# 🔎 Live Query Diagnostics

For every live query, the dashboard can expose the actual search path.

Example:

```text
Query:
"vector database"

nprobe:
11

Selected clusters:
80, 50, 73, 4, 26, ...

Candidates:
6,541 / 50,001

Candidate reduction:
86.9%
```

The dashboard can additionally expose execution stages such as:

```text
Centroid Search
Candidate Gathering
Distance Calculation
Top-K Selection
```

This makes it possible to inspect where the query spends its time instead of treating the vector engine as a black box.

---

# 🎯 Search-Space Reduction

One of Vectra's most important metrics is candidate reduction.

Suppose:

```text
Total vectors:
50,001

IVF candidates:
6,541
```

Then:

\[
CandidateReduction =
1-\frac{6541}{50001}
\]

The dashboard reports this as a percentage.

This answers a more meaningful question than latency alone:

> **How much of the original search space did IVF eliminate?**

Latency is affected by many implementation details, including Python overhead, memory behavior, vectorization, and workload characteristics. Candidate reduction directly exposes the algorithmic effect of partitioning.

---

# 🧠 Result-Level Diagnostics

Vectra's advanced dashboard can explain why an IVF result was considered.

For example:

```text
Result ID: 7906

Similarity: 0.6000

Assigned cluster: #80

Cluster searched: ✓

Reason:
Cluster #80 was among the nearest
centroids selected by nprobe.
```

When an exact result is missed:

```text
⚠ Missed by IVF

Exact rank: #7

Assigned cluster: #63

Cluster searched: ✗

The vector belonged to a cluster that
was not selected by the current nprobe.
```

This provides a concrete explanation of where approximation can introduce recall loss.

---

# 🗑️ Data Lifecycle

Vectra supports logical deletion using tombstones.

The lifecycle is:

```text
INSERT
   │
   ▼
ACTIVE VECTOR
   │
   ▼
SOFT DELETE
   │
   ▼
TOMBSTONE
   │
   ▼
COMPACTION
   │
   ▼
PHYSICALLY PURGED
```

## Why tombstones?

Physical deletion from contiguous vector storage can require expensive data movement.

A tombstone allows deletion to be represented as a logical state change.

Conceptually:

```text
Vector exists
      │
      ▼
Mark deleted
      │
      ▼
Exclude from future searches
```

Physical cleanup can then happen during compaction.

---

# 🛠️ CRUD Operations

Vectra exposes the basic index lifecycle expected from a vector search system:

### Insert

```text
Document
   ↓
Embedding
   ↓
Vector
   ↓
Index insertion
```

### Search

```text
Query
   ↓
Embedding
   ↓
Exact / IVF search
   ↓
Top-K
```

### Delete

```text
Vector ID
   ↓
Tombstone
   ↓
Excluded from search
```

### Compact

```text
Tombstoned entries
        ↓
Remove deleted data
        ↓
Rebuild compact storage
```

All of these operations can be demonstrated through the Streamlit control plane.

---

# 🖥️ Interactive Dashboard

Vectra includes a Streamlit-based visual control plane.

The dashboard is organized around six major areas.

## 🔎 Live Search

Perform natural-language searches and compare:

- Exact search
- IVF-Flat search
- Recall@K
- Exact latency
- IVF latency
- Candidate count
- Candidate reduction
- Top-K
- nprobe

The current query is also shown on the global Pareto frontier.

---

## 📊 Benchmark & Pareto

Visualize the precomputed 500-query benchmark.

Includes:

- Recall@10
- P50 latency
- P95 latency
- Average candidates
- Candidate reduction
- Speedup comparison
- nprobe sweep
- Recall-vs-latency Pareto curve
- Current live-query marker

---

## 🔍 Index Inspector

Inspect the internal IVF structure:

- Number of clusters
- Average cluster size
- Minimum cluster size
- Maximum cluster size
- Cluster distribution
- Vector density across inverted lists
- Current index configuration

---

## 🛠️ Index Operations

Interact directly with the index:

- Insert documents
- Soft-delete vectors
- Inspect tombstones
- Compact the index

This allows the complete data lifecycle to be demonstrated without opening source code.

---

## 📜 Query History

The dashboard maintains session-level search history containing:

- Timestamp
- Query
- Top-K
- nprobe
- Recall@K
- Exact latency
- IVF latency
- Candidate count
- Candidate reduction

This makes it possible to compare multiple live searches.

---

## 💡 How Vectra Works

A visual explanation of:

- Vector embeddings
- Exact search
- K-Means
- IVF-Flat
- nlist
- nprobe
- Candidate selection
- Recall evaluation
- Search complexity

The purpose is to make the internal mechanics of vector search visible rather than hiding them behind an API abstraction.

---

# 📁 Repository Structure

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
│   ├── ground_truth.npy
│   └── benchmark_results.json
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
│   ├── run_benchmarks.py
│   └── demo.py
│
└── tests/
    ├── test_math.py
    ├── test_exact.py
    ├── test_kmeans.py
    ├── test_ivf.py
    ├── test_tombstone.py
    └── test_api.py
```

---

# 🧱 Core Components

| Component | Responsibility |
|---|---|
| `distance.py` | Vector normalization, cosine similarity and vectorized distance operations |
| `exact.py` | Brute-force ground-truth search |
| `kmeans.py` | NumPy-based K-Means clustering |
| `ivf_flat.py` | Handcrafted IVF-Flat index |
| `tombstone.py` | Logical deletion and tombstone tracking |
| `search_service.py` | Search orchestration and index interaction |
| `main.py` | FastAPI REST interface |
| `app.py` | Interactive Streamlit control plane |
| `prepare_data.py` | Dataset and ground-truth preparation |
| `run_benchmarks.py` | Benchmark and evaluation pipeline |
| `demo.py` | End-to-end demonstration |

---

# 📐 Complexity

Let:

- \(N\) = number of vectors
- \(D\) = vector dimension
- \(C\) = number of candidate vectors
- \(L\) = number of IVF clusters
- \(P\) = number of probed clusters
- \(I\) = K-Means iterations

### Exact Search

\[
O(ND)
\]

Every active vector participates in the similarity calculation.

### K-Means

Approximately:

\[
O(INDL)
\]

depending on implementation details and the assignment/recomputation strategy.

### IVF-Flat Query

Conceptually:

```text
Centroid selection
        +
Candidate retrieval
        +
Exact similarity over C candidates
```

The candidate-search component is approximately:

\[
O(CD)
\]

with additional centroid-selection and indexing overhead.

If clusters are approximately uniform:

\[
C \approx P\frac{N}{L}
\]

Therefore the candidate-search portion can be approximated as:

\[
O\left(P\frac{N}{L}D\right)
\]

This is a simplified model; actual latency also depends on centroid search, candidate gathering, memory access, vectorization, and implementation overhead.

### Logical Delete

Tombstone marking is designed to be:

\[
O(1)
\]

### Compaction

Physical compaction requires rebuilding or rewriting active storage and therefore depends on the number of stored vectors and index representation.

---

# 🧪 Testing

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

Individual test modules can also be executed independently:

```bash
python -m pytest tests/test_math.py -v
python -m pytest tests/test_exact.py -v
python -m pytest tests/test_kmeans.py -v
python -m pytest tests/test_ivf.py -v
python -m pytest tests/test_tombstone.py -v
python -m pytest tests/test_api.py -v
```

The test suite covers the major layers of the system:

```text
Vector Mathematics
        ↓
Exact Search
        ↓
K-Means
        ↓
IVF-Flat
        ↓
Tombstones
        ↓
API
```

---

# 🔌 API

The FastAPI service provides a lightweight interface over the search engine.

Start the service:

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

Open interactive API documentation at:

```text
http://localhost:8000/docs
```

The API exposes operations around:

- Health/status
- Search
- Insert
- Delete

The API layer delegates search behavior to the underlying Vectra components rather than implementing a separate search algorithm.

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| NumPy | Vector arithmetic and numerical operations |
| Sentence Transformers | Text-to-vector embedding |
| FastAPI | REST API |
| Streamlit | Interactive dashboard |
| Pytest | Automated testing |

### Deliberately excluded

Vectra does **not** use:

- Pinecone
- FAISS
- Chroma
- `sklearn.neighbors`
- another external vector database

The ANN indexing logic is implemented directly in the project.

---

# 📊 Benchmark Philosophy

Vectra intentionally avoids hard-coded performance claims.

Performance depends on:

- Hardware
- Python version
- NumPy version
- Dataset geometry
- Embedding distribution
- Number of vectors
- Vector dimension
- `nlist`
- `nprobe`
- Query workload
- Memory behavior

Therefore benchmark results should be treated as **measurements of a specific workload and environment**, not universal guarantees.

The benchmark pipeline should be rerun when comparing different implementations or environments.

---

# 🎛️ Important IVF Parameters

## `nlist`

The number of clusters used to partition the vector space.

Higher `nlist` generally means:

```text
More clusters
Smaller average clusters
Potentially fewer candidates per probe
More centroid-selection overhead
```

## `nprobe`

The number of clusters searched for each query.

Higher `nprobe` generally means:

```text
More candidates
Higher recall potential
More distance calculations
Potentially higher latency
```

The interaction between these parameters is one of the main experimental dimensions of Vectra.

---

# 🔍 Exact vs IVF-Flat

| Property | Exact | IVF-Flat |
|---|---|---|
| Search strategy | Brute force | Cluster-based |
| Searches all vectors | Yes | No |
| Distance within candidates | Exact | Exact |
| Approximate? | No | Yes |
| Ground truth | Yes | No |
| Recall | 100% against itself | Measured against exact |
| Main parameter | — | `nprobe` |
| Main trade-off | Computation | Recall vs computation |

The important distinction is:

> **IVF-Flat does not approximate the similarity calculation. It approximates the search space.**

---

# 🗺️ Implementation Roadmap

| Phase | Implementation | Status |
|---|---|---|
| 1 | Project foundation, configuration, logging and metrics | ✅ Complete |
| 2 | Vector mathematics engine | ✅ Complete |
| 3 | Exact brute-force search | ✅ Complete |
| 4 | Corpus embedding and dataset generation | ✅ Complete |
| 5 | Vectorized K-Means | ✅ Complete |
| 6 | Handcrafted IVF-Flat index | ✅ Complete |
| 7 | Recall@K evaluation | ✅ Complete |
| 8 | Latency and performance benchmarking | ✅ Complete |
| 9 | Tombstone soft deletion | ✅ Complete |
| 10 | Search service and FastAPI API | ✅ Complete |
| 11 | Interactive Streamlit control plane | ✅ Complete |
| 12 | End-to-end testing and demonstration | ✅ Complete |

---

# 🧭 Reproducible Workflow

A typical development/evaluation workflow is:

```text
1. Prepare corpus
        │
        ▼
2. Generate embeddings
        │
        ▼
3. Build exact ground truth
        │
        ▼
4. Train K-Means
        │
        ▼
5. Build IVF-Flat
        │
        ▼
6. Run benchmark
        │
        ▼
7. Evaluate Recall@K
        │
        ▼
8. Measure latency
        │
        ▼
9. Visualize Pareto frontier
        │
        ▼
10. Explore live queries
```

Commands:

```bash
python -m scripts.prepare_data
python -m scripts.run_benchmarks
python -m scripts.demo
python -m pytest tests/ -v
python -m streamlit run dashboard/app.py
```

---

# ⚠️ Design Limitations

Vectra is intentionally an in-memory implementation designed to expose vector-search mechanics.

Current limitations include:

- Index state is memory-resident.
- Persistence is not the primary storage model.
- IVF-Flat is the implemented ANN strategy; HNSW is not currently included.
- Tombstones defer physical removal until compaction.
- Benchmark results depend on the execution environment.
- The reference implementation prioritizes transparency and understandability over production-scale distributed deployment.
- The system is not intended to replace mature production vector databases.

These limitations are intentional and make the internal search pipeline easier to inspect.

---

# 🚀 Future Extensions

Possible future directions include:

### HNSW

Implement a navigable small-world graph index from scratch and compare it against IVF-Flat.

### Persistence

Add disk-backed index persistence and recovery.

### WAL

Introduce a write-ahead log for durable mutations.

### Memory Mapping

Support memory-mapped vector storage for larger datasets.

### Batch Search

Optimize multiple-query workloads.

### Adaptive `nprobe`

Select `nprobe` dynamically based on query characteristics and target recall.

### Sharding

Partition the index across multiple workers or machines.

### Quantization

Compare IVF-Flat with compressed representations such as product quantization.

These extensions would turn Vectra from an educational ANN implementation into a broader exploration of vector database architecture.

---

# 🎓 Why Build a Vector Index From Scratch?

Most vector-search applications begin with an API call:

```python
index.search(query)
```

That abstraction is useful, but it hides the underlying engineering.

Vectra intentionally removes that abstraction.

The project exposes:

```text
Embedding
   ↓
Vector
   ↓
Distance
   ↓
Clustering
   ↓
Indexing
   ↓
Candidate Selection
   ↓
Top-K
   ↓
Recall
   ↓
Latency
```

Building the system from first principles makes it possible to understand what vector databases are actually doing underneath their APIs.

The central lesson is simple:

> **Approximate nearest-neighbor search is fundamentally a trade-off between how much of the search space you examine and how much accuracy you retain.**

Vectra makes that trade-off measurable.

---

# 🤝 Contributing

Contributions and experiments are welcome.

Before opening a pull request:

1. Keep the core search algorithms free from external vector-search libraries.
2. Add tests for behavioral changes.
3. Avoid hard-coded benchmark results.
4. Document changes that affect search semantics or evaluation.
5. Run the complete test suite.

```bash
python -m pytest tests/ -v
```

For algorithmic changes, include benchmark comparisons where relevant.

---

# ⭐ Project Summary

Vectra is a from-scratch exploration of vector search:

```text
50,000+ vectors
       │
       ▼
Exact Brute Force
       │
       ▼
Ground Truth
       │
       ▼
K-Means Partitioning
       │
       ▼
IVF-Flat
       │
       ▼
nprobe
       │
       ▼
Candidate Reduction
       │
       ▼
Exact Similarity
       │
       ▼
Top-K
       │
       ├──────────────► Recall@K
       │
       └──────────────► Latency
                              │
                              ▼
                       Pareto Frontier
```

**Vectra: A vector search engine built from scratch to make approximate search understandable, measurable, and inspectable.**
