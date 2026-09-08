import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.service.search_service import search_service
import config

st.set_page_config(
    page_title="Vectra Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Vectra: In-Memory Vector Search Engine")
st.subheader("Interactive Pareto Frontier & Live Retrieval Sandbox")

# 1. Sidebar Configuration & System Metadata
st.sidebar.header("⚙️ Engine Stats")
stats = search_service.get_stats()
st.sidebar.metric("Exact Vector Count", f"{stats['exact'].total_vectors:,}")
st.sidebar.metric("IVF Vector Count", f"{stats['ivf'].total_vectors:,}")
st.sidebar.metric("Vector Dimension", stats['exact'].dimension)

# 2. Benchmark Pareto Visualization Section
benchmark_file = config.DATA_DIR / "benchmark_results.json"

if benchmark_file.exists():
    with open(benchmark_file, "r") as f:
        bench_data = json.load(f)

    st.markdown("---")
    st.markdown("### 📊 Tradeoff Analysis: Recall@10 vs Latency")

    df_sweep = pd.DataFrame(bench_data["ivf_sweep"])
    exact_p50 = bench_data["exact_baseline"]["latency_p50_ms"]

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        fig = px.line(
            df_sweep,
            x="latency_p50_ms",
            y="recall_at_10",
            text="nprobe",
            markers=True,
            labels={"latency_p50_ms": "p50 Latency (ms)", "recall_at_10": "Recall@10"},
            title="IVF-Flat Pareto Curve (nprobe Sweep vs. Brute-Force Baseline)"
        )
        fig.update_traces(textposition="top center", marker=dict(size=10, color="#FF4B4B"))
        fig.add_vline(
            x=exact_p50,
            line_dash="dash",
            line_color="#00CC96",
            annotation_text=f"Exact Baseline ({exact_p50:.2f} ms)"
        )
        st.plotly_chart(fig, width="stretch")

    with col_table:
        st.markdown("**Sweep Metrics Data**")
        st.dataframe(
            df_sweep[["nprobe", "recall_at_10", "latency_p50_ms", "speedup_vs_exact"]],
            hide_index=True,
            width="stretch"
        )

# 3. Interactive Live Search Sandbox
st.markdown("---")
st.markdown("### 🔎 Live Semantic Search Sandbox")

@st.cache_resource
def load_engine():
    search_service.initialize(load_data=True)
    return search_service

engine = load_engine()

col_q, col_k, col_np = st.columns([3, 1, 1])

with col_q:
    user_query = st.text_input(
        "Enter natural language query:",
        "Vector database similarity search with inverted index"
    )
with col_k:
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
with col_np:
    nprobe = st.slider("nprobe (IVF)", min_value=1, max_value=32, value=4)

if st.button("Run Comparative Search", type="primary"):
    raw_texts = []
    if config.TEXT_STORE_PATH.exists():
        with open(config.TEXT_STORE_PATH, "r", encoding="utf-8") as f:
            raw_texts = json.load(f)

    # Search Exact
    res_exact = engine.search(
        query_vector=None,
        query_text=user_query,
        index_type="exact",
        top_k=top_k,
        nprobe=nprobe
    )

    # Search IVF
    res_ivf = engine.search(
        query_vector=None,
        query_text=user_query,
        index_type="ivf",
        top_k=top_k,
        nprobe=nprobe
    )

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.success("🎯 **Exact Search (Ground Truth)**")
        st.metric("Latency", f"{res_exact.latency_ms:.2f} ms")
        st.metric("Candidates Searched", f"{res_exact.candidates_searched:,}")

        for i, (doc_id, score) in enumerate(zip(res_exact.ids, res_exact.scores), 1):
            text_snippet = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
            st.markdown(f"**{i}. ID #{doc_id}** (Similarity: `{score:.4f}`)\n> {text_snippet}")

    with res_col2:
        st.info("⚡ **IVF-Flat Search (ANN)**")
        st.metric("Latency", f"{res_ivf.latency_ms:.2f} ms")
        st.metric("Candidates Searched", f"{res_ivf.candidates_searched:,}")

        for i, (doc_id, score) in enumerate(zip(res_ivf.ids, res_ivf.scores), 1):
            text_snippet = raw_texts[doc_id] if doc_id < len(raw_texts) else f"Vector ID #{doc_id}"
            st.markdown(f"**{i}. ID #{doc_id}** (Similarity: `{score:.4f}`)\n> {text_snippet}")