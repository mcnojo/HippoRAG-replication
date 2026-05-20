import os
import json
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")


def plot_metrics_comparison(results: list[dict], out_dir: str = "figures"):
    # bar chart comparing all metrics side by side
    if not HAS_MATPLOTLIB:
        print("Skipping plot_metrics_comparison: matplotlib not available")
        return

    os.makedirs(out_dir, exist_ok=True)

    labels = [r["label"] for r in results]
    metrics = ["recall@2", "recall@5", "answer_f1", "exact_match"]
    titles = ["Recall@2", "Recall@5", "Answer F1", "Exact Match"]

    x = np.arange(len(metrics))
    n = len(results)
    total_width = 0.7
    bar_width = total_width / n
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    offsets = [bar_width * (i - (n - 1) / 2) for i in range(n)]

    for i, (result, offset) in enumerate(zip(results, offsets)):
        vals = [result.get(m, 0) for m in metrics]
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, vals, bar_width, label=result["label"], color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.set_title("Retrieval and QA Metrics Comparison")
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_retrieval_breakdown(results: list[dict], out_dir: str = "figures"):
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(out_dir, exist_ok=True)

    labels = [r["label"] for r in results]
    metrics = ["recall@2", "recall@5"]
    titles = ["Recall@2", "Recall@5"]

    x = np.arange(len(metrics))
    n = len(results)
    total_width = 0.7
    bar_width = total_width / n
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    offsets = [bar_width * (i - (n - 1) / 2) for i in range(n)]

    for i, (result, offset) in enumerate(zip(results, offsets)):
        vals = [result.get(m, 0) for m in metrics]
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, vals, bar_width, label=result["label"], color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Recall")
    ax.legend()
    ax.set_title("Retrieval Performance")
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "retrieval_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_qa_breakdown(results: list[dict], out_dir: str = "figures"):
    if not HAS_MATPLOTLIB:
        return

    os.makedirs(out_dir, exist_ok=True)

    labels = [r["label"] for r in results]
    metrics = ["answer_f1", "exact_match"]
    titles = ["Answer F1", "Exact Match"]

    x = np.arange(len(metrics))
    n = len(results)
    total_width = 0.7
    bar_width = total_width / n
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    offsets = [bar_width * (i - (n - 1) / 2) for i in range(n)]

    for i, (result, offset) in enumerate(zip(results, offsets)):
        vals = [result.get(m, 0) for m in metrics]
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, vals, bar_width, label=result["label"], color=color)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(titles)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.set_title("Question Answering Performance")
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "qa_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_delta_chart(results: list[dict], out_dir: str = "figures"):
    # horizontal bar chart showing v1->v2 improvement
    if not HAS_MATPLOTLIB or len(results) < 2:
        return

    os.makedirs(out_dir, exist_ok=True)

    metrics = ["recall@2", "recall@5", "answer_f1", "exact_match"]
    titles = ["Recall@2", "Recall@5", "Answer F1", "Exact Match"]

    deltas = [results[1].get(m, 0) - results[0].get(m, 0) for m in metrics]

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]
    bars = ax.barh(titles, deltas, color=colors)

    for bar, delta in zip(bars, deltas):
        width = bar.get_width()
        ax.annotate(f'{delta:+.3f}',
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5 if delta >= 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if delta >= 0 else 'right',
                    va='center',
                    fontsize=10)

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel("Change from v1 to v2")
    ax.set_title("HippoRAG v2 Improvements over v1")
    ax.grid(axis='x', alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "v2_improvement.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def generate_markdown_report(results: list[dict], out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# HippoRAG Benchmark Results",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(r["label"] for r in results) + " |",
        "|--------|" + "|".join(["--------" for _ in results]) + "|",
    ]

    metrics = ["recall@2", "recall@5", "answer_f1", "exact_match"]
    for m in metrics:
        row = f"| {m} |"
        for r in results:
            row += f" {r.get(m, 0):.4f} |"
        lines.append(row)

    lines.extend([
        "",
        "## Details",
        "",
        f"- Questions evaluated: {results[0].get('num_questions', 'N/A')}",
        f"- Top-K retrieved: {results[0].get('top_k', 'N/A')}",
        "",
    ])

    if len(results) >= 2:
        label_a = results[0].get("label", "A")
        label_b = results[1].get("label", "B")
        lines.extend([
            f"## {label_a} vs {label_b} Delta",
            "",
            "| Metric | Delta |",
            "|--------|-------|",
        ])
        for m in metrics:
            delta = results[1].get(m, 0) - results[0].get(m, 0)
            lines.append(f"| {m} | {delta:+.4f} |")

    lines.extend([
        "",
        "## Figures",
        "",
        "![Metrics Comparison](../figures/metrics_comparison.png)",
        "",
        "![Retrieval Comparison](../figures/retrieval_comparison.png)",
        "",
        "![QA Comparison](../figures/qa_comparison.png)",
        "",
        "![V2 Improvement](../figures/v2_improvement.png)",
    ])

    path = os.path.join(out_dir, "benchmark_report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


def plot_results(results_path: str = "results/benchmark_results.json", out_dir: str = "figures"):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        raw = json.load(f)

    # handle both flat list and {"config": ..., "metrics": [...]} formats
    if isinstance(raw, dict) and "metrics" in raw:
        results = raw["metrics"]
    else:
        results = raw

    print(f"Loaded {len(results)} result sets from {results_path}")

    plot_metrics_comparison(results, out_dir)
    plot_retrieval_breakdown(results, out_dir)
    plot_qa_breakdown(results, out_dir)
    plot_delta_chart(results, out_dir)

    results_dir = os.path.dirname(results_path)
    generate_markdown_report(results, results_dir)

    print(f"\nAll figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate HippoRAG benchmark plots")
    parser.add_argument("--results", type=str, default="results/benchmark_results.json")
    parser.add_argument("--output-dir", type=str, default="figures")

    args = parser.parse_args()
    plot_results(args.results, args.output_dir)


if __name__ == "__main__":
    main()
