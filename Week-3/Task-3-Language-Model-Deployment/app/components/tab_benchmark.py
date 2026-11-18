"""
NEXUS NLP — Tab B: Benchmark Dashboard Component
Loads and visualizes the preliminary multi-task evaluation benchmark,
task scorecards, CPU scaling metrics, and individual trial records.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def load_benchmark_data(reports_dir: str):
    """Loads benchmark CSV and JSON files from reports/."""
    json_path = os.path.join(reports_dir, "task_metrics_summary.json")
    csv_path = os.path.join(reports_dir, "benchmark_results.csv")

    metrics_data = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    df_records = pd.DataFrame()
    if os.path.exists(csv_path):
        df_records = pd.read_csv(csv_path)

    return metrics_data, df_records


def render_tab_benchmark(reports_dir: str):
    """Renders Section B: Benchmark Dashboard."""
    st.subheader("Multi-Task Benchmark Dashboard")

    # Preliminary Evaluation Scope Notice
    st.warning(
        "**Preliminary Benchmark Scope & Sample Size Clarification:**\n"
        "The quantitative figures presented below are derived from a curated preliminary evaluation suite "
        "(sample sizes $N=4$ to $6$ per task). These metrics validate end-to-end inference mechanics, verify metric computation "
        "pipelines, and provide directional capability comparisons. **They must not be interpreted as universal or production-grade "
        "performance benchmarks**, which require evaluation on datasets with thousands of instances (e.g. SQuAD, CNN/DM, WMT)."
    )

    metrics_data, df_records = load_benchmark_data(reports_dir)

    if not metrics_data or df_records.empty:
        st.error("Benchmark data not found in reports directory. Please run `python src/benchmark.py` first.")
        return

    # Metric Cards with Sample Sizes
    st.markdown("#### Preliminary Task Scorecard:")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        rouge_l = metrics_data.get("summarization", {}).get("rougeL", 0.0)
        n_sum = metrics_data.get("summarization", {}).get("count", 4)
        st.metric("Summarization", f"{rouge_l:.1f}%", help="ROUGE-L F1 Score (Longest Common Subsequence)")
        st.caption(f"Metric: ROUGE-L | **$N={n_sum}$**")

    with c2:
        qa_f1 = metrics_data.get("question_answering", {}).get("f1", 0.0)
        n_qa = metrics_data.get("question_answering", {}).get("count", 5)
        st.metric("Question Answering", f"{qa_f1:.1f}%", help="Token-level F1 with SQuAD text normalization")
        st.caption(f"Metric: Token F1 | **$N={n_qa}$**")

    with c3:
        cls_acc = metrics_data.get("classification", {}).get("accuracy", 0.0)
        n_cls = metrics_data.get("classification", {}).get("total", 6)
        st.metric("Zero-Shot Classif.", f"{cls_acc:.1f}%", help="Classification accuracy over sentiment, topics, and tone")
        st.caption(f"Metric: Accuracy | **$N={n_cls}$**")

    with c4:
        bleu = metrics_data.get("translation", {}).get("bleu", 0.0)
        n_trans = metrics_data.get("translation", {}).get("count", 5)
        st.metric("Translation", f"{bleu:.1f}", help="Corpus SacreBLEU score against human references")
        st.caption(f"Metric: SacreBLEU | **$N={n_trans}$**")

    with c5:
        reason_acc = metrics_data.get("reasoning", {}).get("accuracy", 0.0)
        n_reason = metrics_data.get("reasoning", {}).get("total", 4)
        st.metric("Reasoning & Logic", f"{reason_acc:.1f}%", help="Qualitative factual correctness on arithmetic and deduction")
        st.caption(f"Metric: Correctness | **$N={n_reason}$**")

    st.markdown("---")

    # Interactive Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("##### Task-wise Performance Comparison ($N=4$ to $6$):")
        task_names = [
            f"Summarization (ROUGE-L, N={n_sum})",
            f"QA (Token F1, N={n_qa})",
            f"Classification (Acc, N={n_cls})",
            f"Translation (BLEU, N={n_trans})",
            f"Reasoning (Acc, N={n_reason})"
        ]
        task_scores = [rouge_l, qa_f1, cls_acc, bleu, reason_acc]
        colors = ["#2563eb", "#0284c7", "#059669", "#d97706", "#7c3aed"]

        fig_tasks = px.bar(
            x=task_names,
            y=task_scores,
            labels={"x": "Task Domain & Sample Size", "y": "Standardized Score (%)"},
            color=task_names,
            color_discrete_sequence=colors,
            text=[f"{s:.1f}%" for s in task_scores]
        )
        fig_tasks.update_traces(textposition="outside", showlegend=False)
        fig_tasks.update_layout(yaxis_range=[0, 110], height=340, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(fig_tasks, width="stretch")

    with col_chart2:
        st.markdown("##### Decoding Strategy: Latency vs. Throughput:")
        # Filter decoding comparison records
        decoding_records = df_records[df_records["task"].str.startswith("decoding_")]
        if not decoding_records.empty:
            agg_dec = decoding_records.groupby("strategy").agg({
                "latency_ms": "mean",
                "tokens_per_second": "mean"
            }).reset_index()

            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(
                x=agg_dec["strategy"],
                y=agg_dec["latency_ms"],
                name="Latency (ms)",
                marker_color="#3b82f6"
            ))
            fig_dec.add_trace(go.Scatter(
                x=agg_dec["strategy"],
                y=agg_dec["tokens_per_second"],
                name="Throughput (tok/s)",
                yaxis="y2",
                mode="lines+markers",
                marker=dict(color="#ea580c", size=8),
                line=dict(color="#ea580c", width=2.5)
            ))
            fig_dec.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=25, b=10),
                yaxis=dict(title="Latency (ms) [Lower is Faster]"),
                yaxis2=dict(title="Throughput (tok/s)", overlaying="y", side="right"),
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_dec, width="stretch")
        else:
            st.info("Decoding records available in detailed trials table.")

    st.markdown("---")

    # Detailed Trials Database
    st.markdown("#### Granular Benchmark Trials Explorer:")
    st.caption("Inspect the 45 raw benchmark runs recorded during Phase 3 CPU evaluation.")

    col_filter_task, col_filter_strategy = st.columns(2)
    with col_filter_task:
        unique_tasks = ["All Tasks"] + sorted(list(df_records["task"].unique()))
        selected_task_filter = st.selectbox("Filter by Task:", unique_tasks)

    with col_filter_strategy:
        unique_strats = ["All Strategies"] + sorted(list(df_records["strategy"].unique()))
        selected_strat_filter = st.selectbox("Filter by Strategy:", unique_strats)

    df_filtered = df_records.copy()
    if selected_task_filter != "All Tasks":
        df_filtered = df_filtered[df_filtered["task"] == selected_task_filter]
    if selected_strat_filter != "All Strategies":
        df_filtered = df_filtered[df_filtered["strategy"] == selected_strat_filter]

    st.dataframe(
        df_filtered[["task", "strategy", "latency_ms", "output_tokens", "tokens_per_second", "generated", "reference"]],
        width="stretch",
        hide_index=True
    )
