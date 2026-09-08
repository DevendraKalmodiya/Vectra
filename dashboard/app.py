import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.service.search_service import search_service
from scripts.run_benchmarks import run_benchmarks
import config

st.set_page_config(
    page_title="Vectra Control Plane",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Engine Initialization (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_engine():
    search_service.initialize(load_data=True)
    return search_service

engine = load_engine()

# Initialize Session State
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_search" not in st.session_state:
    st.session_state.last_search = None

# -----------------------------------------------------------------------------
# Sidebar: System Stats & Controls
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ Vectra Engine")
st.sidebar.caption("In-Memory Vector Search Control Plane")

stats = engine.get_stats()
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")

st.sidebar.success("🟢 Index: Loaded & Active")
st.sidebar.info("🟢 Dataset: 50,000 Vectors (384-d)")

advanced_mode = st.sidebar.toggle("🔬 Advanced Mode", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Engine Metrics")
st.sidebar.metric("Active Vectors", f"{stats['exact'].total_vectors:,}")
st.sidebar.metric("Tombstoned / Deleted", f"{stats['exact'].deleted_vectors:,}")
st.sidebar.metric("IVF Clusters (nlist)", engine.ivf_index.nlist)

# -----------------------------------------------------------------------------
# Header & Global Engine Overview
# -----------------------------------------------------------------------------
st.title("⚡ Vectra: Visual Control Plane")
st.caption("Pure Python / NumPy In-Memory Vector Search Engine ($O(N \\cdot D)$ Brute-Force vs. IVF-Flat ANN)")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Active Vectors", f"{stats['exact'].total_vectors:,}")
m2.metric("Deleted (Tombstones)", f"{stats['exact'].deleted_vectors:,}")
m3.metric("Vector Dimension", f"{stats['exact'].dimension}")
m4.metric("IVF Clusters (nlist)", f"{engine.ivf_index.nlist}")
m5.metric("Index Status", "Trained" if stats['ivf'].is_trained else "Untrained")
m6.metric("Compaction State", "Clean" if stats['exact'].deleted_vectors == 0 else "Needs Compaction")

st.markdown("---")

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_search, tab_pareto, tab_inspector, tab_ops, tab_history, tab_architecture = st.tabs([
    "🔎 Live Search",
    "📊 Benchmark & Pareto",
    "🔍 Index Inspector",
    "🛠️ Index Operations (CRUD)",
    "📜 Query History",
    "💡 How Vectra Works"
])

# Load precomputed text corpus for display
raw_texts = []
if config.TEXT_STORE_PATH.exists():
    with open(config.TEXT_STORE_PATH, "r", encoding="utf-8") as f:
        raw_texts = json.load(f)

# =============================================================================
# TAB 1: LIVE SEMANTIC SEARCH
# =============================================================================
with tab_search:
    st.subheader("🔎 Comparative Semantic Search")
    
    col_q, col_k, col_np = st.columns([3, 1, 1])
    with col_q:
        user_query = st.text_input("Natural Language Query:", "machine learning vector search algorithms")
    with col_k:
        top_k = st.slider("Top-K Results", 1, 20, 10)
    with col_np:
        nprobe = st.slider("nprobe (Cells Searched)", 1, engine.ivf_index.nlist, 4)

    if st.button("🚀 Run Comparative Search", type="primary", width="stretch"):
        # Run Real Searches
        res_exact = engine.search(query_vector=None, query_text=user_query, index_type="exact", top_k=top_k, nprobe=nprobe)
        res_ivf = engine.search(query_vector=None, query_text=user_query, index_type="ivf", top_k=top_k, nprobe=nprobe)

        # Calculate Real Recall@K
        exact_set = set(res_exact.ids[:top_k])
        ivf_set = set(res_ivf.ids[:top_k])
        recall = len(exact_set.intersection(ivf_set)) / len(exact_set) if exact_set else 0.0
        
        total_vectors = stats['exact'].total_vectors
        cand_reduction = (1.0 - (res_ivf.candidates_searched / total_vectors)) * 100.0 if total_vectors > 0 else 0.0

        st.session_state.last_search = {
            "query": user_query,
            "top_k": top_k,
            "nprobe": nprobe,
            "res_exact": res_exact,
            "res_ivf": res_ivf,
            "recall": recall,
            "reduction": cand_reduction
        }

        # Add to Query History
        st.session_state.query_history.insert(0, {
            "Timestamp": time.strftime("%H:%M:%S"),
            "Query": user_query,
            "Top-K": top_k,
            "nprobe": nprobe,
            "Recall@K": f"{recall * 100:.1f}%",
            "Exact Latency": f"{res_exact.latency_ms:.2f} ms",
            "IVF Latency": f"{res_ivf.latency_ms:.2f} ms",
            "Candidates": f"{res_ivf.candidates_searched:,}",
            "Reduction": f"{cand_reduction:.1f}%"
        })

    if st.session_state.last_search:
        ls = st.session_state.last_search
        res_exact = ls["res_exact"]
        res_ivf = ls["res_ivf"]

        st.markdown("### ⚡ Live Query Metrics")
        qm1, qm2, qm3, qm4, qm5 = st.columns(5)
        qm1.metric("Exact Latency", f"{res_exact.latency_ms:.2f} ms")
        qm2.metric("IVF Latency", f"{res_ivf.latency_ms:.2f} ms", delta=f"{res_exact.latency_ms - res_ivf.latency_ms:.2f} ms faster" if res_ivf.latency_ms < res_exact.latency_ms else None)
        qm3.metric("Recall@K", f"{ls['recall'] * 100:.1f}%")
        qm4.metric("Candidates Searched", f"{res_ivf.candidates_searched:,} / {stats['exact'].total_vectors:,}")
        qm5.metric("Candidate Reduction", f"{ls['reduction']:.1f}%")

        # Candidate Reduction Bar
        st.markdown("#### 🎯 Search Space Reduction")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(y=["Exact Search"], x=[stats['exact'].total_vectors], orientation='h', name="Full Space", marker_color="#00CC96"))
        fig_bar.add_trace(go.Bar(y=["IVF-Flat"], x=[res_ivf.candidates_searched], orientation='h', name="Pruned Candidates", marker_color="#FF4B4B"))
        fig_bar.update_layout(barmode='overlay', height=180, margin=dict(l=20, r=20, t=20, b=20), showlegend=True)
        st.plotly_chart(fig_bar, width="stretch")

        # Side-by-Side Results Comparison
        col_ex, col_ivf = st.columns(2)
        
        with col_ex:
            st.success("🎯 **Exact Search (Ground Truth Baseline)**")
            st.caption(f"Scanned {res_exact.candidates_searched:,} vectors | Latency: {res_exact.latency_ms:.2f} ms")
            
            exact_ids = res_exact.ids
            for rank, (doc_id, score) in enumerate(zip(res_exact.ids, res_exact.scores), 1):
                text_snip = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
                st.markdown(f"**#{rank} | ID #{doc_id}** (Similarity: `{score:.4f}`)\n> {text_snip}")

        with col_ivf:
            st.info("⚡ **IVF-Flat Search (Approximate Nearest Neighbor)**")
            st.caption(f"Scanned {res_ivf.candidates_searched:,} candidates | Latency: {res_ivf.latency_ms:.2f} ms")

            exact_id_set = set(exact_ids)
            for rank, (doc_id, score) in enumerate(zip(res_ivf.ids, res_ivf.scores), 1):
                text_snip = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
                match_status = "✅ True Positive" if doc_id in exact_id_set else "❌ False Positive / Miss"
                st.markdown(f"**#{rank} | ID #{doc_id}** [{match_status}] (Similarity: `{score:.4f}`)\n> {text_snip}")

        # Advanced Mode Diagnostics
        if advanced_mode:
            st.markdown("---")
            st.markdown("### 🔬 Advanced Query Diagnostics & Internal Execution")
            
            col_diag1, col_diag2 = st.columns(2)
            with col_diag1:
                st.markdown("**IVF Selected Clusters (nprobe nearest centroids)**")
                st.write(f"Searched Cells: `{res_ivf.selected_clusters}`")
                
            with col_diag2:
                st.markdown("**Query Execution Stage Timing Breakdown**")
                df_stages = pd.DataFrame(list(res_ivf.stage_latencies_ms.items()), columns=["Pipeline Stage", "Latency (ms)"])
                st.dataframe(df_stages, hide_index=True, width="stretch")

            with st.expander("🔎 'Why Did This Result Match?' (Vector Cluster Assignments)"):
                for doc_id in res_ivf.ids:
                    cluster_id = engine.get_vector_cluster(doc_id)
                    selected = "Yes (Scanned)" if cluster_id in res_ivf.selected_clusters else "No"
                    st.write(f"- **Vector ID #{doc_id}**: Assigned to Cluster `{cluster_id}` | Cluster Selected: `{selected}`")

# =============================================================================
# TAB 2: BENCHMARK & PARETO FRONTIER
# =============================================================================
with tab_pareto:
    st.subheader("📊 Ground Truth Recall@10 vs. Latency Pareto Curve")
    
    benchmark_file = config.DATA_DIR / "benchmark_results.json"
    
    col_bench_btn, col_bench_info = st.columns([1, 3])
    with col_bench_btn:
        if st.button("⚡ Run Full 500-Query Benchmark Suite", type="primary"):
            with st.spinner("Running precomputed ground-truth benchmark across 500 queries..."):
                run_benchmarks()
            st.rerun()

    if benchmark_file.exists():
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)

        df_sweep = pd.DataFrame(bench_data["ivf_sweep"])
        exact_p50 = bench_data["exact_baseline"]["latency_p50_ms"]

        fig_pareto = px.line(
            df_sweep, x="latency_p50_ms", y="recall_at_10", text="nprobe", markers=True,
            labels={"latency_p50_ms": "p50 Latency (ms)", "recall_at_10": "Recall@10"},
            title="Global IVF-Flat Pareto Curve (Sweep over nprobe values)"
        )
        fig_pareto.update_traces(textposition="top center", marker=dict(size=10, color="#FF4B4B"))
        fig_pareto.add_vline(x=exact_p50, line_dash="dash", line_color="#00CC96", annotation_text=f"Exact Baseline ({exact_p50:.2f} ms)")

        # Plot live query overlay point if available
        if st.session_state.last_search:
            ls = st.session_state.last_search
            fig_pareto.add_trace(go.Scatter(
                x=[ls["res_ivf"].latency_ms],
                y=[ls["recall"]],
                mode="markers+text",
                name="Current Live Query",
                text=[f"Live (nprobe={ls['nprobe']})"],
                textposition="bottom center",
                marker=dict(size=16, color="#FFA500", symbol="star")
            ))

        st.plotly_chart(fig_pareto, width="stretch")

        st.markdown("#### 📋 Global Benchmark Results")
        st.dataframe(df_sweep, hide_index=True, width="stretch")

# =============================================================================
# TAB 3: INDEX INSPECTOR
# =============================================================================
with tab_inspector:
    st.subheader("🔍 IVF-Flat Inverted Index Inspector")
    
    cluster_sizes = engine.get_cluster_sizes()
    df_clusters = pd.DataFrame(list(cluster_sizes.items()), columns=["Cluster ID", "Vector Count"])

    ci1, ci2, ci3, ci4 = st.columns(4)
    ci1.metric("Total Clusters (nlist)", len(cluster_sizes))
    ci2.metric("Average Cluster Size", f"{df_clusters['Vector Count'].mean():.1f}")
    ci3.metric("Min Cluster Size", f"{df_clusters['Vector Count'].min()}")
    ci4.metric("Max Cluster Size", f"{df_clusters['Vector Count'].max()}")

    st.markdown("#### 📊 Vector Distribution Across IVF Voronoi Cells")
    st.caption("Balanced clusters ensure predictable candidate set sizes and consistent search speedups.")
    
    fig_hist = px.bar(df_clusters, x="Cluster ID", y="Vector Count", title="Vector Density per Inverted List Cluster")
    st.plotly_chart(fig_hist, width="stretch")

# =============================================================================
# TAB 4: INDEX OPERATIONS (CRUD)
# =============================================================================
with tab_ops:
    st.subheader("🛠️ Dynamic Index Operations & Tombstone Compaction")

    col_ins, col_del = st.columns(2)

    with col_ins:
        st.markdown("### ➕ Insert New Vector / Document")
        ins_id = st.number_input("Vector ID:", min_value=100000, max_value=999999, value=100001)
        ins_text = st.text_area("Document Text to Embed:", "Vectra is a high-performance in-memory vector database.")
        
        if st.button("Insert Document into Engine"):
            engine.insert(vector_id=ins_id, text=ins_text)
            st.success(f"Successfully embedded and inserted Vector ID #{ins_id}!")
            st.rerun()

    with col_del:
        st.markdown("### 🗑️ Soft-Delete Vector (Tombstone)")
        del_id = st.number_input("Vector ID to Delete:", min_value=0, max_value=999999, value=0)
        
        if st.button("Soft-Delete Vector"):
            deleted = engine.delete(vector_id=del_id)
            if deleted:
                st.warning(f"Vector ID #{del_id} soft-deleted with tombstone bitmask.")
                st.rerun()
            else:
                st.error(f"Vector ID #{del_id} not found or already deleted.")

    st.markdown("---")
    st.markdown("### 🧹 Memory Compaction & Tombstone Purging")
    
    comp_col1, comp_col2 = st.columns([2, 1])
    with comp_col1:
        st.write(f"- Active Vectors: `{stats['exact'].total_vectors:,}`")
        st.write(f"- Soft-Deleted Tombstones: `{stats['exact'].deleted_vectors:,}`")
        st.caption("Soft-deletions use bitmask flags for instant $O(1)$ operations. Compaction purges soft-deleted entries from contiguous arrays.")
    
    with comp_col2:
        if st.button("⚡ Compact Index Now", type="primary"):
            p_ex, p_ivf = engine.compact()
            st.success(f"Compaction complete! Purged {p_ex} exact vectors and {p_ivf} IVF vector entries.")
            st.rerun()

# =============================================================================
# TAB 5: QUERY HISTORY
# =============================================================================
with tab_history:
    st.subheader("📜 Session Query History")
    
    if st.session_state.query_history:
        df_hist = pd.DataFrame(st.session_state.query_history)
        st.dataframe(df_hist, hide_index=True, width="stretch")
    else:
        st.info("No queries executed in this session yet. Run a search in the 'Live Search' tab.")

# =============================================================================
# TAB 6: HOW VECTRA WORKS
# =============================================================================
with tab_architecture:
    st.subheader("💡 Vectra Architecture & Query Execution Pipeline")
    
    st.markdown("""
```text
                     ┌──────────────────────────────┐
                     │     Natural Language Query   │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   all-MiniLM-L6-v2 Model     │
                     │  (384-Dim Query Vector q)    │
                     └──────────────┬───────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
┌──────────────────────────────┐                ┌──────────────────────────────┐
│       Exact Index            │                │      IVF-Flat Index          │
│   (Ground Truth Baseline)    │                │  (Approximate Nearest Neighbor)│
├──────────────────────────────┤                ├──────────────────────────────┤
│ 1. Scan ALL N vectors        │                │ 1. Find nprobe nearest       │
│ 2. Compute Cosine Sim        │                │    centroids via K-Means    │
│ 3. Select Exact Top-K        │                │ 2. Gather candidate vectors  │
│                              │                │    from assigned cells       │
│ Time Complexity: O(N · D)    │                │ 3. Compute Cosine Sim on     │
└──────────────┬───────────────┘                │    candidates ONLY           │
               │                                │ 4. Select Approximate Top-K  │
               │                                │                              │
               │                                │ Time Complexity: O(nprobe · N/nlist · D)│
               │                                └──────────────┬───────────────┘
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Recall@K & Speedup Metric  │
                        │   Evaluation Dashboard       │
                        └──────────────────────────────┘

```
""")