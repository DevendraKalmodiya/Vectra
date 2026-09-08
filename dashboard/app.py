import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.service.search_service import search_service
from scripts.run_benchmarks import run_benchmarks
import config

# -----------------------------------------------------------------------------
# Streamlit Page & Theme Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Vectra Control Plane",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Navigation Tabs, Metric Cards, and Result Badges
st.markdown("""
<style>
    /* Metric Card Custom Container */
    div[data-testid="stMetric"] {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #89B4FA !important;
        white-space: nowrap;
        overflow: visible;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #CDD6F4 !important;
        font-weight: 600;
    }
    /* Tab Navigation Enhancements */
    button[data-baseweb="tab"] {
        background-color: #181825;
        border: 1px solid #313244;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        margin-right: 4px;
        font-weight: 600;
        color: #A6ADC8;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #1E1E2E !important;
        border-top: 3px solid #89B4FA !important;
        border-bottom: none !important;
        color: #89B4FA !important;
    }
    /* Result Card Styling */
    .result-card {
        background-color: #181825;
        border-left: 4px solid #313244;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 4px;
    }
    .result-card-exact { border-left-color: #00CC96; }
    .result-card-ivf { border-left-color: #89B4FA; }
    .badge-tp {
        background-color: #1b4332;
        color: #52b788;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .badge-fp {
        background-color: #4a0e17;
        color: #f72585;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Engine Initialization & Session State Setup
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    search_service.initialize(load_data=True)
    return search_service

engine = load_engine()

if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_search" not in st.session_state:
    st.session_state.last_search = None

stats = engine.get_stats()
exact_stats = stats["exact"]
ivf_stats = stats["ivf"]

# Calculate storage counts
total_allocated = exact_stats.total_vectors + exact_stats.deleted_vectors
active_vectors = exact_stats.total_vectors
tombstones = exact_stats.deleted_vectors

# -----------------------------------------------------------------------------
# Sidebar: Engine Status & Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ VECTRA ENGINE")
st.sidebar.caption("Visual Control Plane & Vector Database Inspection")

st.sidebar.markdown("### 🟢 System Status")
st.sidebar.markdown("- **Index State**: Loaded & Active")
st.sidebar.markdown("- **Dataset**: Precomputed Embeddings")
st.sidebar.markdown("- **Embedding Model**: `all-MiniLM-L6-v2`")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Index Overview")
st.sidebar.metric("Active Vectors", f"{active_vectors:,}")
st.sidebar.metric("Tombstones", f"{tombstones:,}")
st.sidebar.metric("Allocated Entries", f"{total_allocated:,}")
st.sidebar.metric("Vector Dimension", f"{exact_stats.dimension}")
st.sidebar.metric("IVF Clusters (nlist)", f"{engine.ivf_index.nlist}")

st.sidebar.markdown("---")
advanced_mode = st.sidebar.toggle("⚙️ Advanced Mode", value=False, help="Expose internal execution timing breakdowns, centroid cluster selection, and vector assignments.")

# -----------------------------------------------------------------------------
# Main Header & System Overview
# -----------------------------------------------------------------------------
st.title("⚡ Vectra: Visual Control Plane")
st.caption("In-Memory Vector Search Engine ($O(N \\cdot D)$ Brute-Force Exact Search vs. IVF-Flat Voronoi ANN)")

st.markdown("### 🏛️ System Overview")
ov1, ov2, ov3, ov4, ov5, ov6, ov7 = st.columns(7)
ov1.metric("Active Vectors", f"{active_vectors:,}")
ov2.metric("Total Allocated", f"{total_allocated:,}")
ov3.metric("Tombstones", f"{tombstones:,}")
ov4.metric("Dimension", f"{exact_stats.dimension}")
ov5.metric("nlist (Clusters)", f"{engine.ivf_index.nlist}")
ov6.metric("Index Status", "Trained" if ivf_stats.is_trained else "Untrained")
ov7.metric("Compaction", "Clean State" if tombstones == 0 else f"{tombstones} Pending")

st.markdown("---")

# -----------------------------------------------------------------------------
# Tab Navigation Structure
# -----------------------------------------------------------------------------
tab_search, tab_pareto, tab_inspector, tab_ops, tab_history, tab_architecture = st.tabs([
    "🔎 Live Search",
    "📊 Benchmark & Pareto",
    "🔍 Index Inspector",
    "🛠️ Index Operations",
    "📜 Query History",
    "💡 How Vectra Works"
])

# Load raw text store for displaying document snippets
raw_texts = []
if config.TEXT_STORE_PATH.exists():
    with open(config.TEXT_STORE_PATH, "r", encoding="utf-8") as f:
        raw_texts = json.load(f)

# =============================================================================
# TAB 1: LIVE SEMANTIC SEARCH (HERO FEATURE)
# =============================================================================
with tab_search:
    st.subheader("🔎 Live Semantic Search Sandbox")
    st.caption("Compare brute-force exact cosine search against approximate IVF-Flat cell scanning in real-time.")

    # Search Controls Box
    with st.container():
        col_q, col_k, col_np, col_btn = st.columns([3, 1, 1, 1.2])
        with col_q:
            user_query = st.text_input("Natural Language Query:", "machine learning vector search algorithms")
        with col_k:
            top_k = st.number_input("Top-K", min_value=1, max_value=50, value=10)
        with col_np:
            nprobe = st.number_input("nprobe", min_value=1, max_value=engine.ivf_index.nlist, value=4)
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_search = st.button("⚡ Run Comparative Search", type="primary", use_container_width=True)

    # Current Configuration Indicator
    st.info(f"**Current Search Configuration**: Top-K = `{top_k}` | nprobe = `{nprobe}` | nlist = `{engine.ivf_index.nlist}` | Active Search Space = `{active_vectors:,}` vectors")

    if run_search or st.session_state.last_search is None:
        res_exact = engine.search(query_vector=None, query_text=user_query, index_type="exact", top_k=top_k, nprobe=nprobe)
        res_ivf = engine.search(query_vector=None, query_text=user_query, index_type="ivf", top_k=top_k, nprobe=nprobe)

        # Calculate exact Recall@K
        exact_set = set(res_exact.ids[:top_k])
        ivf_set = set(res_ivf.ids[:top_k])
        recall = len(exact_set.intersection(ivf_set)) / len(exact_set) if exact_set else 0.0
        
        cand_reduction = (1.0 - (res_ivf.candidates_searched / active_vectors)) * 100.0 if active_vectors > 0 else 0.0

        st.session_state.last_search = {
            "query": user_query,
            "top_k": top_k,
            "nprobe": nprobe,
            "res_exact": res_exact,
            "res_ivf": res_ivf,
            "recall": recall,
            "reduction": cand_reduction
        }

        # Session Query History Log
        st.session_state.query_history.insert(0, {
            "Timestamp": time.strftime("%H:%M:%S"),
            "Query": user_query,
            "Top-K": top_k,
            "nprobe": nprobe,
            "Recall@K": recall,
            "Exact Latency (ms)": round(res_exact.latency_ms, 2),
            "IVF Latency (ms)": round(res_ivf.latency_ms, 2),
            "Candidates Searched": res_ivf.candidates_searched,
            "Space Reduction %": round(cand_reduction, 1)
        })

    ls = st.session_state.last_search
    res_exact = ls["res_exact"]
    res_ivf = ls["res_ivf"]

    # Key Insight Sentence Banner
    st.markdown(
        f"#### 💡 Live Query Insight\n"
        f"> **IVF-Flat evaluated `{res_ivf.candidates_searched:,}` of `{active_vectors:,}` vectors "
        f"— an `{ls['reduction']:.1f}%` search space reduction — while recovering `{ls['recall'] * 100:.1f}%` of the exact ground-truth top-K results.**"
    )

    st.markdown("---")
    st.markdown("### 📊 Current Query Performance")
    qp1, qp2, qp3, qp4, qp5 = st.columns(5)
    qp1.metric("Exact Latency", f"{res_exact.latency_ms:.2f} ms")
    qp2.metric("IVF-Flat Latency", f"{res_ivf.latency_ms:.2f} ms", delta=f"{res_exact.latency_ms - res_ivf.latency_ms:.2f} ms" if res_ivf.latency_ms < res_exact.latency_ms else "0.00 ms")
    qp3.metric("Recall@K Accuracy", f"{ls['recall'] * 100:.1f}%")
    qp4.metric("Candidates Evaluated", f"{res_ivf.candidates_searched:,}")
    qp5.metric("Candidate Reduction", f"{ls['reduction']:.1f}%")

    # Search Space Reduction Bar Chart
    st.markdown("#### 🎯 Search Space Comparison")
    fig_space = go.Figure()
    fig_space.add_trace(go.Bar(
        y=["Exact Search (Brute Force)", "IVF-Flat (Cell Pruned)"],
        x=[active_vectors, res_ivf.candidates_searched],
        orientation='h',
        text=[f"{active_vectors:,} vectors (100%)", f"{res_ivf.candidates_searched:,} candidates ({100 - ls['reduction']:.1f}%)"],
        textposition='auto',
        marker_color=["#00CC96", "#89B4FA"]
    ))
    fig_space.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Vectors Evaluated",
        showlegend=False
    )
    st.plotly_chart(fig_space, use_container_width=True)

    # Side-by-Side Results Comparison
    st.markdown("### ⚔️ Exact vs. IVF-Flat Retrieval Results")
    col_ex_res, col_ivf_res = st.columns(2)

    exact_ids_order = res_exact.ids
    exact_id_set = set(exact_ids_order)

    with col_ex_res:
        st.success("🎯 **Exact Search (Ground Truth Baseline)**")
        st.caption(f"Evaluated all {res_exact.candidates_searched:,} vectors | Latency: {res_exact.latency_ms:.2f} ms")
        
        for rank, (doc_id, score) in enumerate(zip(res_exact.ids, res_exact.scores), 1):
            text_snippet = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
            st.markdown(
                f"<div class='result-card result-card-exact'>"
                f"<b>Rank #{rank} | Vector ID #{doc_id}</b> — Cosine Similarity: <code>{score:.4f}</code><br/>"
                f"<span style='color: #A6ADC8; font-size: 0.9rem;'>{text_snippet}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    with col_ivf_res:
        st.info("⚡ **IVF-Flat Search (Approximate Nearest Neighbor)**")
        st.caption(f"Evaluated {res_ivf.candidates_searched:,} candidates in {nprobe} cells | Latency: {res_ivf.latency_ms:.2f} ms")

        for rank, (doc_id, score) in enumerate(zip(res_ivf.ids, res_ivf.scores), 1):
            text_snippet = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
            is_tp = doc_id in exact_id_set
            badge_html = "<span class='badge-tp'>✓ True Positive</span>" if is_tp else "<span class='badge-fp'>❌ False Positive / Lower Rank</span>"
            
            st.markdown(
                f"<div class='result-card result-card-ivf'>"
                f"<b>Rank #{rank} | Vector ID #{doc_id}</b> {badge_html} — Cosine Similarity: <code>{score:.4f}</code><br/>"
                f"<span style='color: #A6ADC8; font-size: 0.9rem;'>{text_snippet}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    # Advanced Mode Diagnostics Section
    if advanced_mode:
        st.markdown("---")
        st.markdown("### 🔬 Advanced Query Diagnostics & Engine Internals")
        
        col_diag_clusters, col_diag_timing = st.columns(2)
        with col_diag_clusters:
            st.markdown("**Selected IVF Voronoi Cells (Nearest Centroids)**")
            st.write(f"- $nprobe$ Cells Searched: `{len(res_ivf.selected_clusters)}` / `{engine.ivf_index.nlist}`")
            st.write(f"- Centroid Cluster IDs: `{res_ivf.selected_clusters}`")
            st.write(f"- Total Candidate Offsets Gathered: `{res_ivf.candidates_searched:,}`")

        with col_diag_timing:
            st.markdown("**Query Execution Stage Timing Breakdown**")
            if res_ivf.stage_latencies_ms:
                df_timing = pd.DataFrame(
                    list(res_ivf.stage_latencies_ms.items()),
                    columns=["Pipeline Stage", "Measured Latency (ms)"]
                )
                st.dataframe(df_timing, hide_index=True, use_container_width=True)
            else:
                st.caption("Stage timing breakdown not available.")

        with st.expander("🔎 'Why Did This Result Match?' (Match & Miss Diagnostics)"):
            st.markdown("#### Retrived IVF Candidate Diagnostics")
            for doc_id in res_ivf.ids:
                cid = engine.get_vector_cluster(doc_id)
                was_searched = cid in res_ivf.selected_clusters
                st.write(f"- **Vector ID #{doc_id}**: Assigned to Cluster `#{cid}` | Cluster Searched: `{'✓ Yes' if was_searched else '✗ No'}`")

            # Miss Analysis
            missed_ids = [vid for vid in exact_ids_order if vid not in ivf_set]
            if missed_ids:
                st.markdown("#### ⚠ Exact Ground-Truth Items Omitted by IVF")
                for missed_id in missed_ids:
                    cid = engine.get_vector_cluster(missed_id)
                    st.write(f"- **Exact ID #{missed_id}**: Assigned to Cluster `#{cid}` | Cluster Searched: `✗ No (Cell bypassed by nprobe={nprobe})`")

# =============================================================================
# TAB 2: BENCHMARK & PARETO FRONTIER
# =============================================================================
with tab_pareto:
    st.subheader("📊 Ground Truth Evaluation & Pareto Frontier Analysis")
    st.caption("Precomputed 500-query evaluation sweep benchmarking trade-offs between speedup and Recall@10.")

    benchmark_file = config.DATA_DIR / "benchmark_results.json"
    
    col_bench_run, col_bench_opt = st.columns([1, 2])
    with col_bench_run:
        if st.button("⚡ Execute 500-Query Benchmark Suite", type="primary"):
            with st.spinner("Evaluating ground truth across 500 query vectors..."):
                run_benchmarks()
            st.rerun()

    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)

        df_sweep = pd.DataFrame(bench_data["ivf_sweep"])
        exact_p50 = bench_data["exact_baseline"]["latency_p50_ms"]
        exact_p95 = bench_data["exact_baseline"]["latency_p95_ms"]

        st.markdown("---")
        st.markdown("### 📈 Global IVF-Flat Pareto Curve (Recall@10 vs. p50 Latency)")
        
        fig_pareto = px.line(
            df_sweep,
            x="latency_p50_ms",
            y="recall_at_10",
            text="nprobe",
            markers=True,
            labels={"latency_p50_ms": "p50 Latency (ms)", "recall_at_10": "Recall@10 Accuracy"},
            title="Global Benchmark Sweep (500 Queries)"
        )
        fig_pareto.update_traces(textposition="top center", marker=dict(size=10, color="#FF4B4B"))
        fig_pareto.add_vline(
            x=exact_p50,
            line_dash="dash",
            line_color="#00CC96",
            annotation_text=f"Exact Baseline p50 ({exact_p50:.2f} ms)"
        )

        # Overlay Current Live Query Marker
        if st.session_state.last_search:
            ls = st.session_state.last_search
            fig_pareto.add_trace(go.Scatter(
                x=[ls["res_ivf"].latency_ms],
                y=[ls["recall"]],
                mode="markers+text",
                name="Current Live Query",
                text=[f"Live Query (nprobe={ls['nprobe']})"],
                textposition="bottom center",
                marker=dict(size=16, color="#FFA500", symbol="star")
            ))

        st.plotly_chart(fig_pareto, use_container_width=True)

        # Optimal Operating Point Selector
        st.markdown("---")
        st.markdown("### 🎯 Target Recall Operating Point Selector")
        target_recall = st.slider("Select Target Recall Threshold (%):", 50.0, 100.0, 95.0, 1.0) / 100.0
        
        # Find lowest latency config meeting target recall
        valid_configs = df_sweep[df_sweep["recall_at_10"] >= target_recall]
        if not valid_configs.empty:
            best_config = valid_configs.sort_values(by="latency_p50_ms").iloc[0]
            st.success(
                f"**Optimal Configuration for Target Recall ≥ {target_recall * 100:.0f}%**:\n"
                f"- **Recommended nprobe**: `{int(best_config['nprobe'])}` cells\n"
                f"- **Achieved Recall@10**: `{best_config['recall_at_10'] * 100:.2f}%`\n"
                f"- **p50 Latency**: `{best_config['latency_p50_ms']:.2f} ms` (Speedup: `{best_config['speedup_vs_exact']:.1f}x` vs. Exact)\n"
                f"- **Average Candidates Evaluated**: `{int(best_config['avg_candidates_searched']):,}` vectors"
            )
        else:
            st.warning("No nprobe configuration met the target recall threshold.")

        # Benchmark Metrics Table
        st.markdown("---")
        st.markdown("### 📋 Precomputed Global Benchmark Metrics")
        st.dataframe(df_sweep, hide_index=True, use_container_width=True)

        # Latency Distribution Comparison Chart
        st.markdown("#### ⏱️ Latency Percentile Comparison (p50 vs. p95)")
        fig_lat = go.Figure()
        fig_lat.add_trace(go.Bar(x=df_sweep["nprobe"].astype(str), y=df_sweep["latency_p50_ms"], name="p50 Latency (ms)", marker_color="#89B4FA"))
        fig_lat.add_trace(go.Bar(x=df_sweep["nprobe"].astype(str), y=df_sweep["latency_p95_ms"], name="p95 Latency (ms)", marker_color="#F38BA8"))
        fig_lat.update_layout(xaxis_title="nprobe setting", yaxis_title="Latency (ms)", barmode='group', height=300)
        st.plotly_chart(fig_lat, use_container_width=True)

# =============================================================================
# TAB 3: INDEX INSPECTOR
# =============================================================================
with tab_inspector:
    st.subheader("🔍 IVF-Flat Inverted Index Inspector")
    st.caption("Inspect structural vector distribution across K-Means Voronoi partitions.")

    cluster_sizes = engine.get_cluster_sizes()
    df_clusters = pd.DataFrame(list(cluster_sizes.items()), columns=["Cluster ID", "Vector Density"])

    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("Total Clusters (nlist)", len(cluster_sizes))
    ci2.metric("Mean Cluster Size", f"{df_clusters['Vector Density'].mean():.1f}")
    ci3.metric("Min Cluster Size", f"{df_clusters['Vector Density'].min():,}")
    ci4.metric("Max Cluster Size", f"{df_clusters['Vector Density'].max():,}")

    st.markdown("---")
    st.markdown("### 📊 Vector Distribution Across IVF Clusters")
    st.caption("Note: Natural variance in cluster sizes directly impacts candidate set sizes and query latency predictability.")

    fig_hist = px.bar(
        df_clusters,
        x="Cluster ID",
        y="Vector Density",
        title="Vector Counts per Inverted List Partition",
        color="Vector Density",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# TAB 4: INDEX OPERATIONS (CRUD)
# =============================================================================
with tab_ops:
    st.subheader("🛠️ Index Mutation & Memory Compaction Controls")

    # Visual Storage Lifecycle Banner
    st.markdown("#### 🔄 Storage Vector Lifecycle")
    st.info("`INSERT` ➔ **ACTIVE** ➔ `DELETE` ➔ **TOMBSTONE** ➔ `COMPACT` ➔ **PURGED & RECLAIMED**")

    col_ins, col_del = st.columns(2)

    with col_ins:
        st.markdown("### ➕ Insert Document / Vector")
        ins_id = st.number_input("Vector ID:", min_value=100000, max_value=999999, value=100001)
        ins_text = st.text_area("Document Content (Embedded on-the-fly):", "Vectra provides vector search with custom inverted indexes.")
        
        if st.button("Insert Document into Engine"):
            engine.insert(vector_id=ins_id, text=ins_text)
            st.success(f"✓ Inserted Vector ID #{ins_id} successfully!")
            st.rerun()

    with col_del:
        st.markdown("### 🗑️ Soft-Delete Vector")
        del_id = st.number_input("Vector ID to Delete:", min_value=0, max_value=999999, value=0)
        
        if st.button("Soft-Delete Vector"):
            deleted = engine.delete(vector_id=del_id)
            if deleted:
                st.warning(f"✓ Soft-deleted Vector ID #{del_id} with tombstone bitmask.")
                st.rerun()
            else:
                st.error(f"Vector ID #{del_id} not found or already soft-deleted.")

    st.markdown("---")
    st.markdown("### 🧹 Tombstone Compaction Manager")
    
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.write(f"- **Active Vectors**: `{active_vectors:,}`")
        st.write(f"- **Soft-Deleted Tombstones**: `{tombstones:,}`")
        st.write(f"- **Total Allocated Entries**: `{total_allocated:,}`")
        st.caption("Soft deletions use $O(1)$ bitmask markers. Compaction purges soft-deleted entries from contiguous arrays.")

    with cc2:
        if st.button("⚡ Compact Index Now", type="primary"):
            p_ex, p_ivf = engine.compact()
            st.success(f"Compaction complete! Purged {p_ex} exact vectors and {p_ivf} IVF entries.")
            st.rerun()

# =============================================================================
# TAB 5: QUERY HISTORY
# =============================================================================
with tab_history:
    st.subheader("📜 Current Session Query History")

    if st.session_state.query_history:
        df_hist = pd.DataFrame(st.session_state.query_history)
        
        # Summary Aggregate Metrics
        hs1, hs2, hs3, hs4 = st.columns(4)
        hs1.metric("Queries Executed", len(df_hist))
        hs2.metric("Mean Recall@K", f"{df_hist['Recall@K'].mean() * 100:.1f}%")
        hs3.metric("Mean IVF Latency", f"{df_hist['IVF Latency (ms)'].mean():.2f} ms")
        hs4.metric("Mean Candidate Reduction", f"{df_hist['Space Reduction %'].mean():.1f}%")

        st.markdown("---")
        st.dataframe(df_hist, hide_index=True, use_container_width=True)
    else:
        st.info("No queries executed in this session yet. Run a search in the 'Live Search' tab.")

# =============================================================================
# TAB 6: HOW VECTRA WORKS
# =============================================================================
with tab_architecture:
    st.subheader("💡 Architecture & Search Execution Pipeline")

    st.markdown("""
### 1. Vector Search Paradigm

* **Exact Brute-Force Search**: Scans all $N$ vectors in contiguous memory to compute exact Cosine similarity ($O(N \\cdot D)$). Provides exact ground truth.
* **IVF-Flat Approximate Search**: Partitions vector space into $nlist$ Voronoi cells via K-Means. At query time, finds $nprobe$ nearest centroids and evaluates exact distances **only within candidate cells** ($O(nprobe \\cdot \\frac{N}{nlist} \\cdot D)$).
* **Flat Distance Math**: Within selected candidate cells, standard $L_2$ normalized inner-product distance is computed without lossy vector quantization.

---

### 2. Pipeline Execution Flow

```text
                        ┌──────────────────────────────┐
                        │    Natural Language Query    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    all-MiniLM-L6-v2 Model    │
                        │   (384-Dim Query Vector q)   │
                        └──────────────┬───────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │                                                     │
            ▼                                                     ▼
┌──────────────────────────────┐                      ┌──────────────────────────────┐
│        Exact Index           │                      │       IVF-Flat Index         │
│   (Ground Truth Baseline)    │                      │(Approximate Nearest Neighbor)│
├──────────────────────────────┤                      ├──────────────────────────────┤
│ 1. Scan ALL N vectors        │                      │ 1. Predict nprobe nearest    │
│ 2. Compute Cosine Sim        │                      │    centroids via K-Means     │
│ 3. Select Exact Top-K        │                      │ 2. Gather candidate vectors  │
│                              │                      │    from assigned inverted list│
│ Time Complexity: O(N · D)    │                      │ 3. Compute Cosine Sim on     │
└──────────────┬───────────────┘                      │    candidates ONLY           │
               │                                      │ 4. Select Approximate Top-K  │
               │                                      │                              │
               │                                      │ Time Complexity:             │
               │                                      │ O(nprobe · N/nlist · D)      │
               │                                      └──────────────┬───────────────┘
               │                                                     │
               └───────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Recall@K & Speedup Metric  │
                        │     Evaluation Control       │
                        └──────────────────────────────┘
```
""")