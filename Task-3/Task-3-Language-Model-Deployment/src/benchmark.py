"""
NEXUS NLP — Multi-Task Evaluation Benchmark & Profiling Harness
Executes quantitative evaluation across Summarization, QA, Classification,
Translation, and Reasoning; benchmarks decoding strategies; and produces
publication-grade figures and CSV reports.

Author: Subham Gupta (ShadowFox ML Engineer Intern)
Task 3 - Language Model Deployment & Intelligence Evaluation Lab
"""

import os
import sys

# Ensure Task-3 root is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import time
import json
import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.model_engine import NexusModelEngine
from src.metrics import (
    compute_rouge,
    compute_sacrebleu,
    evaluate_qa,
    compute_classification_metrics,
    normalize_text
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NexusBenchmark")

# ==============================================================================
# Canonical Multi-Task Evaluation Benchmark Dataset
# ==============================================================================

BENCHMARK_DATA = {
    "summarization": [
        {
            "prompt": "Summarize the following article: Deep learning has revolutionized artificial intelligence over the past decade. By training deep neural networks on massive datasets with gradient-based optimization algorithms, models have achieved super-human capabilities in image classification, protein structure prediction, and natural language understanding. However, training these foundational architectures demands immense computational power, specialized tensor processing hardware, and substantial electrical energy consumption.",
            "reference": "Deep learning has transformed AI through training deep neural networks on large datasets, achieving super-human performance in vision, biology, and language, despite high computational and energy requirements."
        },
        {
            "prompt": "Summarize: Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose molecules. Water and carbon dioxide are transformed into sugar and oxygen using solar photons captured by chlorophyll within chloroplasts.",
            "reference": "Photosynthesis is the process where plants convert light, water, and carbon dioxide into glucose and oxygen using chlorophyll."
        },
        {
            "prompt": "Summarize: The James Webb Space Telescope was launched in December 2021 as NASA's premier space observatory. Operating at the Second Sun-Earth Lagrange Point 1.5 million kilometers away, it uses ultra-sensitive infrared sensors and an origami-style gold-coated beryllium mirror to observe the earliest stars and galaxies formed after the Big Bang.",
            "reference": "The James Webb Space Telescope is an advanced infrared observatory launched in 2021 to observe the earliest stars and galaxies from the L2 point."
        },
        {
            "prompt": "Summarize: Cloud computing provides on-demand access to computing resources over the internet, including storage, servers, databases, and software applications. Businesses utilize cloud platforms to eliminate capital expenditures for physical infrastructure, scale resources dynamically according to demand, and accelerate software deployment cycles globally.",
            "reference": "Cloud computing offers scalable, on-demand internet access to servers, storage, and software, reducing capital expenses and accelerating software deployment."
        }
    ],
    "question_answering": [
        {
            "prompt": "Context: The Apollo 11 mission was the first spaceflight that landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Question: Who was the commander of the Apollo 11 mission?",
            "reference": "Neil Armstrong"
        },
        {
            "prompt": "Context: Python was created in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands as a successor to the ABC programming language. Question: Who created the Python programming language?",
            "reference": "Guido van Rossum"
        },
        {
            "prompt": "Context: The Transformer architecture was introduced in the seminal 2017 paper 'Attention Is All You Need' by Vaswani et al. from Google Brain and Google Research. Question: In what year was the Transformer architecture introduced?",
            "reference": "2017"
        },
        {
            "prompt": "Context: The mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in a small molecule called adenosine triphosphate (ATP). Question: What molecule stores chemical energy produced by mitochondria?",
            "reference": "adenosine triphosphate"
        },
        {
            "prompt": "Context: Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China–Nepal border runs across its summit point. Question: What mountain range contains Mount Everest?",
            "reference": "Himalayas"
        }
    ],
    "classification": [
        {
            "prompt": "Classify the sentiment of this text as positive or negative: The customer service was astonishingly responsive and resolved my technical issue within minutes.",
            "reference": "positive",
            "allowed_labels": ["positive", "negative"]
        },
        {
            "prompt": "Classify the sentiment of this text as positive or negative: The package arrived three weeks late, damaged, and customer support was completely unhelpful.",
            "reference": "negative",
            "allowed_labels": ["positive", "negative"]
        },
        {
            "prompt": "Classify the news category as Technology, Sports, or Finance: The Federal Reserve raised benchmark interest rates by twenty-five basis points to combat inflation pressures.",
            "reference": "Finance",
            "allowed_labels": ["Technology", "Sports", "Finance"]
        },
        {
            "prompt": "Classify the news category as Technology, Sports, or Finance: The star striker scored a sensational hat-trick in stoppage time to secure the championship trophy.",
            "reference": "Sports",
            "allowed_labels": ["Technology", "Sports", "Finance"]
        },
        {
            "prompt": "Classify the news category as Technology, Sports, or Finance: Researchers unveiled a new 2-nanometer semiconductor manufacturing process that improves energy efficiency by forty percent.",
            "reference": "Technology",
            "allowed_labels": ["Technology", "Sports", "Finance"]
        },
        {
            "prompt": "Classify the tone of this statement as Formal or Informal: We are writing to formally acknowledge receipt of your correspondence dated March 14th.",
            "reference": "Formal",
            "allowed_labels": ["Formal", "Informal"]
        }
    ],

    "translation": [
        {
            "prompt": "Translate English to French: Artificial intelligence is transforming modern science and healthcare.",
            "reference": "L'intelligence artificielle transforme la science et la santé modernes."
        },
        {
            "prompt": "Translate English to French: Good morning, welcome to our university research symposium.",
            "reference": "Bonjour, bienvenue à notre symposium de recherche universitaire."
        },
        {
            "prompt": "Translate English to German: The weather forecast predicts sunshine and pleasant temperatures for the weekend.",
            "reference": "Die Wettervorhersage sagt Sonnenschein und angenehme Temperaturen für das Wochenende voraus."
        },
        {
            "prompt": "Translate English to German: Open source software encourages international scientific collaboration.",
            "reference": "Open-Source-Software fördert die internationale wissenschaftliche Zusammenarbeit."
        },
        {
            "prompt": "Translate English to Spanish: Machine learning models require clean, well-curated training data.",
            "reference": "Los modelos de aprendizaje automático requieren datos de entrenamiento limpios y bien seleccionados."
        }
    ],
    "reasoning": [
        {
            "prompt": "Solve this arithmetic problem: If a factory produces 25 units per hour and operates for 8 hours a day, how many units does it produce in 5 days? Answer with the final number:",
            "reference": "1000",
            "type": "Arithmetic"
        },
        {
            "prompt": "Reason logically: All mammals are warm-blooded. All dolphins are mammals. Are dolphins warm-blooded? Answer yes or no:",
            "reference": "yes",
            "type": "Deductive Logic"
        },
        {
            "prompt": "Solve this riddle: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
            "reference": "echo",
            "type": "Abductive Reasoning"
        },
        {
            "prompt": "Answer with the correct choice: If tomorrow is Wednesday, what day was yesterday? (A) Monday (B) Tuesday (C) Thursday (D) Friday. Answer:",
            "reference": "Monday",
            "type": "Temporal Reasoning"
        }
    ]
}


# ==============================================================================
# Benchmark Runner
# ==============================================================================

class NexusBenchmarkRunner:
    def __init__(self, engine: NexusModelEngine):
        self.engine = engine
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.reports_dir = os.path.join(self.base_dir, "reports")
        self.figures_dir = os.path.join(self.reports_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        self.results_records: List[Dict[str, Any]] = []

    def run_multi_task_evaluation(self) -> Dict[str, Any]:
        """Runs the standard evaluation suite across all 5 tasks."""
        logger.info("Starting Multi-Task Quantitative Evaluation ...")
        task_metrics = {}

        # 1. Summarization
        logger.info("Evaluating Summarization (ROUGE-1, ROUGE-2, ROUGE-L) ...")
        sum_preds, sum_refs = [], []
        for item in BENCHMARK_DATA["summarization"]:
            res = self.engine.generate(item["prompt"], strategy="beam_search", num_beams=4, max_new_tokens=64)
            sum_preds.append(res.generated_text)
            sum_refs.append(item["reference"])
            self.results_records.append({
                "task": "summarization",
                "prompt": item["prompt"],
                "generated": res.generated_text,
                "reference": item["reference"],
                "strategy": "beam_search",
                "latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_second": res.tokens_per_second,
                "metric_name": "ROUGE",
                "primary_score": None  # Populated after batch scoring
            })
        sum_scores = compute_rouge(sum_preds, sum_refs)
        task_metrics["summarization"] = sum_scores
        logger.info(f"Summarization: ROUGE-1={sum_scores['rouge1']:.2f}, ROUGE-2={sum_scores['rouge2']:.2f}, ROUGE-L={sum_scores['rougeL']:.2f}")

        # 2. Question Answering
        logger.info("Evaluating Question Answering (Exact Match, Token F1) ...")
        qa_preds, qa_refs = [], []
        for item in BENCHMARK_DATA["question_answering"]:
            res = self.engine.generate(item["prompt"], strategy="greedy", max_new_tokens=32)
            qa_preds.append(res.generated_text)
            qa_refs.append(item["reference"])
            self.results_records.append({
                "task": "question_answering",
                "prompt": item["prompt"],
                "generated": res.generated_text,
                "reference": item["reference"],
                "strategy": "greedy",
                "latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_second": res.tokens_per_second,
                "metric_name": "ExactMatch_F1",
                "primary_score": None
            })
        qa_scores = evaluate_qa(qa_preds, qa_refs)
        task_metrics["question_answering"] = qa_scores
        logger.info(f"Question Answering: Exact Match={qa_scores['exact_match']}%, Token F1={qa_scores['f1']}%")

        # 3. Classification
        logger.info("Evaluating Zero-Shot Classification (Accuracy) ...")
        cls_preds, cls_refs, cls_allowed = [], [], []
        for item in BENCHMARK_DATA["classification"]:
            res = self.engine.generate(item["prompt"], strategy="greedy", max_new_tokens=16)
            cls_preds.append(res.generated_text)
            cls_refs.append(item["reference"])
            cls_allowed.append(item.get("allowed_labels"))
            self.results_records.append({
                "task": "classification",
                "prompt": item["prompt"],
                "generated": res.generated_text,
                "reference": item["reference"],
                "strategy": "greedy",
                "latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_second": res.tokens_per_second,
                "metric_name": "Accuracy",
                "primary_score": None
            })
        cls_scores = compute_classification_metrics(cls_preds, cls_refs, allowed_labels=cls_allowed)
        task_metrics["classification"] = cls_scores
        logger.info(f"Classification: Accuracy={cls_scores['accuracy']}% ({cls_scores['correct']}/{cls_scores['total']}) | Macro-F1={cls_scores.get('macro_f1', 0.0)}%")


        # 4. Translation
        logger.info("Evaluating Machine Translation (SacreBLEU) ...")
        trans_preds, trans_refs = [], []
        for item in BENCHMARK_DATA["translation"]:
            res = self.engine.generate(item["prompt"], strategy="beam_search", num_beams=4, max_new_tokens=48)
            trans_preds.append(res.generated_text)
            trans_refs.append(item["reference"])
            self.results_records.append({
                "task": "translation",
                "prompt": item["prompt"],
                "generated": res.generated_text,
                "reference": item["reference"],
                "strategy": "beam_search",
                "latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_second": res.tokens_per_second,
                "metric_name": "SacreBLEU",
                "primary_score": None
            })
        trans_scores = compute_sacrebleu(trans_preds, trans_refs)
        task_metrics["translation"] = trans_scores
        logger.info(f"Translation: SacreBLEU={trans_scores['bleu']}")

        # 5. Reasoning (Instruction Following & Logic)
        logger.info("Evaluating Reasoning & Instruction Following ...")
        reason_preds, reason_refs = [], []
        for item in BENCHMARK_DATA["reasoning"]:
            res = self.engine.generate(item["prompt"], strategy="greedy", max_new_tokens=32)
            reason_preds.append(res.generated_text)
            reason_refs.append(item["reference"])
            self.results_records.append({
                "task": "reasoning",
                "prompt": item["prompt"],
                "generated": res.generated_text,
                "reference": item["reference"],
                "strategy": "greedy",
                "latency_ms": res.latency_ms,
                "input_tokens": res.input_tokens,
                "output_tokens": res.output_tokens,
                "tokens_per_second": res.tokens_per_second,
                "metric_name": "QualitativeCorrectness",
                "primary_score": None
            })
        reason_scores = compute_classification_metrics(reason_preds, reason_refs)
        task_metrics["reasoning"] = reason_scores
        logger.info(f"Reasoning: Accuracy={reason_scores['accuracy']}% ({reason_scores['correct']}/{reason_scores['total']})")

        return task_metrics

    def run_decoding_strategy_experiments(self) -> pd.DataFrame:
        """
        Runs controlled parameter variation across identical inputs:
        - Greedy
        - Beam Search (num_beams=2, 4)
        - Nucleus Sampling (T=0.7, p=0.9; T=1.2, p=0.95)
        - Repetition Penalty (1.0 vs 1.5 vs 2.0)
        """
        logger.info("Running Decoding Strategy Comparison Experiments ...")

        test_prompts = [
            ("Summarize: The James Webb Space Telescope was launched in December 2021 as NASA's premier space observatory. Operating at the Second Sun-Earth Lagrange Point 1.5 million kilometers away, it uses ultra-sensitive infrared sensors and an origami-style gold-coated beryllium mirror to observe the earliest stars and galaxies formed after the Big Bang.", "Summarization"),
            ("Translate English to French: Artificial intelligence is transforming modern science and healthcare.", "Translation"),
            ("Write an intuitive analogy explaining why neural networks require multiple layers of non-linear activations:", "Creative/Open"),
        ]

        configs = [
            {"name": "Greedy", "strategy": "greedy", "kwargs": {"max_new_tokens": 48}},
            {"name": "Beam Search (k=2)", "strategy": "beam_search", "kwargs": {"num_beams": 2, "max_new_tokens": 48}},
            {"name": "Beam Search (k=4)", "strategy": "beam_search", "kwargs": {"num_beams": 4, "max_new_tokens": 48}},
            {"name": "Sampling (T=0.7, p=0.9)", "strategy": "sampling", "kwargs": {"temperature": 0.7, "top_p": 0.9, "top_k": 50, "max_new_tokens": 48}},
            {"name": "Sampling (T=1.2, p=0.95)", "strategy": "sampling", "kwargs": {"temperature": 1.2, "top_p": 0.95, "top_k": 50, "max_new_tokens": 48}},
            {"name": "Repetition Penalty (1.5)", "strategy": "greedy", "kwargs": {"repetition_penalty": 1.5, "max_new_tokens": 48}},
            {"name": "Repetition Penalty (2.0)", "strategy": "greedy", "kwargs": {"repetition_penalty": 2.0, "max_new_tokens": 48}},
        ]

        decoding_records = []
        for prompt_text, prompt_category in test_prompts:
            for cfg in configs:
                res = self.engine.generate(
                    prompt=prompt_text,
                    strategy=cfg["strategy"],
                    **cfg["kwargs"]
                )
                decoding_records.append({
                    "experiment": "decoding_comparison",
                    "config_name": cfg["name"],
                    "strategy": cfg["strategy"],
                    "category": prompt_category,
                    "prompt": prompt_text,
                    "generated_text": res.generated_text,
                    "latency_ms": res.latency_ms,
                    "output_tokens": res.output_tokens,
                    "tokens_per_second": res.tokens_per_second,
                    "memory_rss_mb": res.memory_rss_mb
                })
                # Add to global records as well
                self.results_records.append({
                    "task": f"decoding_{prompt_category.lower()}",
                    "prompt": prompt_text,
                    "generated": res.generated_text,
                    "reference": "N/A",
                    "strategy": cfg["name"],
                    "latency_ms": res.latency_ms,
                    "input_tokens": res.input_tokens,
                    "output_tokens": res.output_tokens,
                    "tokens_per_second": res.tokens_per_second,
                    "metric_name": "Latency_Throughput",
                    "primary_score": res.tokens_per_second
                })

        df_decoding = pd.DataFrame(decoding_records)
        return df_decoding

    def run_latency_scaling_profile(self) -> pd.DataFrame:
        """
        Profiles CPU latency scaling across variable output lengths:
        max_new_tokens = [16, 32, 64, 96, 128]
        """
        logger.info("Running CPU Latency & Throughput Scaling Profile ...")
        prompt = "Summarize the history, development, and architectural milestones of modern natural language processing and transformer models in detail."

        token_lengths = [16, 32, 64, 96, 128]
        profiling_records = []

        for length in token_lengths:
            # Run 3 trials per length and take median for rigorous statistical stability
            trial_latencies = []
            trial_tps = []
            for _ in range(3):
                res = self.engine.generate(prompt, strategy="greedy", max_new_tokens=length)
                trial_latencies.append(res.latency_ms)
                trial_tps.append(res.tokens_per_second)

            med_latency = float(np.median(trial_latencies))
            med_tps = float(np.median(trial_tps))

            profiling_records.append({
                "max_new_tokens": length,
                "median_latency_ms": round(med_latency, 2),
                "median_tokens_per_second": round(med_tps, 2),
                "ms_per_token": round(med_latency / length, 2)
            })

        df_profile = pd.DataFrame(profiling_records)
        return df_profile

    def save_reports_and_visualizations(
        self,
        task_metrics: Dict[str, Any],
        df_decoding: pd.DataFrame,
        df_profile: pd.DataFrame
    ) -> None:
        """Generates publication-quality charts and exports benchmark_results.csv."""
        logger.info("Saving CSV reports and generating publication-quality figures ...")

        # 1. Save benchmark_results.csv
        csv_path = os.path.join(self.reports_dir, "benchmark_results.csv")
        df_all = pd.DataFrame(self.results_records)
        df_all.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved benchmark results CSV to: {csv_path} ({len(df_all)} records)")

        # Save task metrics JSON
        json_path = os.path.join(self.reports_dir, "task_metrics_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(task_metrics, f, indent=2)

        # Style setup for publication charts
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 300
        })

        # Chart 1: Task-wise Primary Performance Metrics
        plt.figure(figsize=(9, 5))
        tasks = ["Summarization\n(ROUGE-L)", "Question Answering\n(Token F1)", "Classification\n(Accuracy)", "Translation\n(SacreBLEU)", "Reasoning\n(Correctness)"]
        scores = [
            task_metrics["summarization"]["rougeL"],
            task_metrics["question_answering"]["f1"],
            task_metrics["classification"]["accuracy"],
            task_metrics["translation"]["bleu"],
            task_metrics["reasoning"]["accuracy"]
        ]
        colors = ["#2b5c8f", "#3c8dbc", "#2ca02c", "#d95f02", "#7570b3"]
        bars = plt.bar(tasks, scores, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 1.5, f"{height:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

        plt.ylim(0, 110)
        plt.title("NEXUS NLP — Multi-Task Intelligence Benchmark (FLAN-T5-small)", pad=15, fontweight="bold")
        plt.ylabel("Standardized Metric Score (%)", labelpad=10)
        plt.tight_layout()
        chart1_path = os.path.join(self.figures_dir, "task_performance_summary.png")
        plt.savefig(chart1_path, dpi=300)
        plt.close()
        logger.info(f"Generated chart: {chart1_path}")

        # Chart 2: Decoding Strategy Comparison (Latency vs Throughput)
        plt.figure(figsize=(10, 5.5))
        decoding_agg = df_decoding.groupby("config_name").agg({
            "latency_ms": "mean",
            "tokens_per_second": "mean"
        }).reset_index().sort_values("latency_ms")

        x = np.arange(len(decoding_agg))
        width = 0.35

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        rects1 = ax1.bar(x - width/2, decoding_agg["latency_ms"], width, label="Mean Latency (ms)", color="#3470a3", edgecolor="black", linewidth=0.6)
        rects2 = ax2.bar(x + width/2, decoding_agg["tokens_per_second"], width, label="Throughput (Tokens/s)", color="#e67e22", edgecolor="black", linewidth=0.6)

        ax1.set_xlabel("Decoding Strategy Configuration", labelpad=10, fontweight="bold")
        ax1.set_ylabel("Inference Latency (ms) [Lower is Faster]", color="#3470a3", labelpad=10, fontweight="bold")
        ax2.set_ylabel("Throughput (Tokens / Sec) [Higher is Faster]", color="#e67e22", labelpad=10, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(decoding_agg["config_name"], rotation=25, ha="right")
        plt.title("NEXUS NLP — Decoding Strategy Latency vs. Throughput Benchmark", pad=15, fontweight="bold")

        fig.tight_layout()
        chart2_path = os.path.join(self.figures_dir, "decoding_strategy_comparison.png")
        plt.savefig(chart2_path, dpi=300)
        plt.close()
        logger.info(f"Generated chart: {chart2_path}")

        # Chart 3: CPU Latency Scaling vs Output Length
        plt.figure(figsize=(8.5, 4.8))
        plt.plot(df_profile["max_new_tokens"], df_profile["median_latency_ms"], marker="o", color="#d62728", linewidth=2.5, markersize=8, label="Actual Median CPU Latency")

        # Fit linear trendline to demonstrate O(N) autoregressive decoder scaling
        z = np.polyfit(df_profile["max_new_tokens"], df_profile["median_latency_ms"], 1)
        p = np.poly1d(z)
        plt.plot(df_profile["max_new_tokens"], p(df_profile["max_new_tokens"]), linestyle="--", color="#555555", alpha=0.8, label=f"Linear Trend (Slope: {z[0]:.2f} ms/tok)")

        for _, row in df_profile.iterrows():
            plt.annotate(
                f"{row['median_latency_ms']:.0f} ms\n({row['ms_per_token']:.1f} ms/tok)",
                (row["max_new_tokens"], row["median_latency_ms"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                fontweight="bold"
            )

        plt.title("NEXUS NLP — Autoregressive Decoder Latency Scaling on CPU", pad=15, fontweight="bold")
        plt.xlabel("Target Maximum New Tokens", labelpad=10)
        plt.ylabel("Inference Latency (ms)", labelpad=10)
        plt.ylim(0, max(df_profile["median_latency_ms"]) * 1.25)
        plt.legend(loc="upper left")
        plt.tight_layout()
        chart3_path = os.path.join(self.figures_dir, "latency_scaling_analysis.png")
        plt.savefig(chart3_path, dpi=300)
        plt.close()
        logger.info(f"Generated chart: {chart3_path}")

        # Chart 4: Token Throughput Distribution
        plt.figure(figsize=(8, 4.5))
        all_tps = [r["tokens_per_second"] for r in self.results_records if r.get("tokens_per_second", 0) > 0]
        sns.histplot(all_tps, kde=True, color="#2ca02c", bins=12, edgecolor="black", alpha=0.6)
        mean_tps = np.mean(all_tps)
        plt.axvline(mean_tps, color="#d62728", linestyle="--", linewidth=2, label=f"Mean Throughput: {mean_tps:.1f} tok/s")

        plt.title("NEXUS NLP — CPU Token Generation Throughput Distribution", pad=15, fontweight="bold")
        plt.xlabel("Generation Throughput (Tokens / Second)", labelpad=10)
        plt.ylabel("Execution Count", labelpad=10)
        plt.legend(loc="upper right")
        plt.tight_layout()
        chart4_path = os.path.join(self.figures_dir, "throughput_distribution.png")
        plt.savefig(chart4_path, dpi=300)
        plt.close()
        logger.info(f"Generated chart: {chart4_path}")


def main():
    logger.info("Initializing NEXUS Benchmark Suite ...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, "model_cache")

    engine = NexusModelEngine(cache_dir=cache_dir, device="cpu")
    engine.load()

    runner = NexusBenchmarkRunner(engine)
    task_metrics = runner.run_multi_task_evaluation()
    df_decoding = runner.run_decoding_strategy_experiments()
    df_profile = runner.run_latency_scaling_profile()

    runner.save_reports_and_visualizations(task_metrics, df_decoding, df_profile)
    logger.info("Benchmark & Profiling Suite completed successfully!")


if __name__ == "__main__":
    main()
