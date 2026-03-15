"""
eval/visualize.py

Reads evaluation_results.csv and generates a comprehensive visual report:
  - Bar chart of mean scores per metric
  - Per-sample line chart for sample-level trends
  - Color-coded annotation for pass/fail thresholds

Saves → eval/ragas_evaluation_report.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

_ROOT    = Path(__file__).resolve().parent.parent
CSV_FILE = _ROOT / "eval" / "evaluation_results_final.csv"
OUT_IMG  = _ROOT / "eval" / "ragas_evaluation_report.png"

METRIC_COLORS = {
    "faithfulness":     "#4CAF50",
    "answer_relevancy": "#2196F3",
    "context_precision": "#FF9800",
    "context_recall":   "#9C27B0",
}
PASS_THRESHOLD = 0.6   # Scores above this are considered acceptable


def generate_report():
    if not CSV_FILE.exists():
        print(f"Results file not found: {CSV_FILE}")
        return

    df = pd.read_csv(CSV_FILE)
    metrics = [m for m in METRIC_COLORS.keys() if m in df.columns]

    if not metrics:
        print("No recognised Ragas metric columns found in CSV.")
        return

    means = df[metrics].mean()
    n_samples = len(df)

    # Layout: top row = summary cards + bar chart; bottom row = per-sample lines
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("RAG Pipeline Evaluation Report", fontsize=18, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Summary metric cards (top-left) ────────────────────────────────
    ax_cards = fig.add_subplot(gs[0, 0])
    ax_cards.axis("off")
    card_y = 0.92
    for m, score in means.items():
        color  = METRIC_COLORS.get(m, "#607D8B")
        status = "✔ PASS" if score >= PASS_THRESHOLD else "✘ FAIL"
        status_color = "#388E3C" if score >= PASS_THRESHOLD else "#D32F2F"
        ax_cards.text(0.03, card_y, f"{m.replace('_',' ').title()}", fontsize=10, fontweight="bold", color=color, transform=ax_cards.transAxes)
        ax_cards.text(0.70, card_y, f"{score:.3f}  {status}", fontsize=10, color=status_color, transform=ax_cards.transAxes)
        card_y -= 0.18
    ax_cards.text(0.03, card_y - 0.05, f"Samples: {n_samples}", fontsize=9, color="gray", transform=ax_cards.transAxes)
    ax_cards.set_title("Metric Summary", fontsize=12, fontweight="bold")

    # ── 2. Aggregated bar chart (top-right) ───────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1])
    bars = ax_bar.barh(
        [m.replace("_", " ").title() for m in means.index],
        means.values,
        color=[METRIC_COLORS.get(m, "#607D8B") for m in means.index],
        edgecolor="white", height=0.55
    )
    ax_bar.axvline(PASS_THRESHOLD, color="#E53935", linestyle="--", linewidth=1.5, label=f"Threshold ({PASS_THRESHOLD})")
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_xlabel("Score (0.0 – 1.0)", fontsize=10)
    ax_bar.set_title("Average Metric Scores", fontsize=12, fontweight="bold")
    ax_bar.legend(fontsize=9)
    for bar, val in zip(bars, means.values):
        ax_bar.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

    # ── 3. Per-sample line chart (bottom, both columns) ───────────────────
    ax_line = fig.add_subplot(gs[1, :])
    x = np.arange(n_samples)
    for m in metrics:
        color = METRIC_COLORS.get(m, "#607D8B")
        ax_line.plot(x, df[m].fillna(0), marker="o", label=m.replace("_", " ").title(), color=color, linewidth=1.8, markersize=5)
    ax_line.axhline(PASS_THRESHOLD, color="#E53935", linestyle="--", linewidth=1.2, alpha=0.7, label=f"Threshold ({PASS_THRESHOLD})")
    ax_line.set_xticks(x)
    ax_line.set_xticklabels([f"Q{i+1}" for i in x], fontsize=9)
    ax_line.set_ylim(-0.05, 1.1)
    ax_line.set_ylabel("Score", fontsize=10)
    ax_line.set_xlabel("Evaluation Sample", fontsize=10)
    ax_line.set_title("Per-Sample Score Breakdown", fontsize=12, fontweight="bold")
    ax_line.legend(fontsize=9, loc="lower right")
    ax_line.grid(axis="y", linestyle="--", alpha=0.4)

    plt.savefig(OUT_IMG, dpi=180, bbox_inches="tight")
    print(f"✅ Visual report saved → {OUT_IMG}")


if __name__ == "__main__":
    generate_report()
