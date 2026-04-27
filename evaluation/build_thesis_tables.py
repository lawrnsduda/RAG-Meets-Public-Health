"""
Generate thesis-ready tables and figures from evaluation JSONs.

After the Colab notebook has finished, copy the results from Drive
to a local folder and run:

    python build_thesis_tables.py path/to/results/

This script will create:
- tables/  -> Markdown tables ready to paste into Word
- figures/ -> PNG plots for confusion matrices and selective tradeoff
- summary.md -> consolidated overview document

Required input files in the results folder:
- eval_qwen3_*.json
- eval_mistral_*.json
- hallucination_qwen3_*.json
- hallucination_mistral_*.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


CONDITIONS_ORDER = [
    "Baseline",
    "Baseline+Abstain",
    "RAG-only",
    "RAG-only+Abstain",
    "RAG+NLI",
    "RAG+NLI+Abstain",
]

LLMS = ["qwen3", "mistral"]
LABELS_3 = ["SUPPORTED", "REFUTED", "MISLEADING"]
LABELS_4 = LABELS_3 + ["NOT_ENOUGH_EVIDENCE"]


def find_latest(results_dir: Path, pattern: str) -> Path | None:
    matches = sorted(results_dir.glob(pattern))
    return matches[-1] if matches else None


def load_eval_data(results_dir: Path) -> dict:
    """Load latest eval_*.json for each LLM."""
    out = {}
    for llm in LLMS:
        path = find_latest(results_dir, f"eval_{llm}_*.json")
        if path is None:
            print(f"  WARNING: no eval file for {llm} found")
            continue
        with open(path) as f:
            out[llm] = json.load(f)
        print(f"  loaded eval: {path.name}")
    return out


def load_hallucination_data(results_dir: Path) -> dict:
    """Load latest hallucination_*.json for each LLM."""
    out = {}
    for llm in LLMS:
        path = find_latest(results_dir, f"hallucination_{llm}_*.json")
        if path is None:
            print(f"  WARNING: no hallucination file for {llm} found")
            continue
        with open(path) as f:
            out[llm] = json.load(f)
        print(f"  loaded hallucination: {path.name}")
    return out


# Table 1: Overall classification performance per condition x LLM
def table_overall_performance(eval_data: dict, out_dir: Path) -> None:
    """Accuracy + Macro-F1 per condition per LLM."""
    lines = ["| Condition | Qwen3 Acc | Qwen3 F1 | Mistral Acc | Mistral F1 |",
             "|---|---|---|---|---|"]

    for cond in CONDITIONS_ORDER:
        row = [cond]
        for llm in LLMS:
            data = eval_data.get(llm, {}).get(cond, {})
            metrics = data.get("metrics", {})
            acc = metrics.get("accuracy")
            f1 = metrics.get("macro_f1")
            row.append(f"{acc:.3f}" if acc is not None else "-")
            row.append(f"{f1:.3f}" if f1 is not None else "-")
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    out = "\n".join(lines)
    (out_dir / "table_overall_performance.md").write_text(out)
    print("Wrote table_overall_performance.md")


# Table 2: Per-class F1 (3-class label set, all conditions)
def table_per_class_f1(eval_data: dict, out_dir: Path) -> None:
    """F1 per class per condition per LLM."""
    lines = []
    for llm in LLMS:
        lines.append(f"\n### {llm.title()}\n")
        lines.append("| Condition | F1-SUPPORTED | F1-REFUTED | F1-MISLEADING |")
        lines.append("|---|---|---|---|")

        for cond in CONDITIONS_ORDER:
            data = eval_data.get(llm, {}).get(cond, {})
            per_class = data.get("metrics", {}).get("per_class", {})
            row = [cond]
            for label in LABELS_3:
                f1 = per_class.get(label, {}).get("f1")
                row.append(f"{f1:.3f}" if f1 is not None else "-")
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    out = "\n".join(lines)
    (out_dir / "table_per_class_f1.md").write_text(out)
    print("Wrote table_per_class_f1.md")


# Table 3: Selective Classification metrics (only abstain conditions)
def table_selective(eval_data: dict, out_dir: Path) -> None:
    """Coverage, selective accuracy, abstention by class."""
    lines = []
    for llm in LLMS:
        lines.append(f"\n### {llm.title()}\n")
        lines.append("| Condition | Coverage | Selective Acc | Abst-SUP | Abst-REF | Abst-MIS |")
        lines.append("|---|---|---|---|---|---|")

        for cond in CONDITIONS_ORDER:
            if not cond.endswith("+Abstain"):
                continue
            data = eval_data.get(llm, {}).get(cond, {})
            sel = data.get("metrics", {}).get("selective", {})
            cov = sel.get("coverage")
            sel_acc = sel.get("selective_accuracy")
            abst = sel.get("abstention_by_class", {})

            row = [cond]
            row.append(f"{cov:.3f}" if cov is not None else "-")
            row.append(f"{sel_acc:.3f}" if sel_acc is not None else "-")
            for label in LABELS_3:
                a = abst.get(label)
                row.append(f"{a:.3f}" if a is not None else "-")
            lines.append("| " + " | ".join(row) + " |")

    out = "\n".join(lines)
    (out_dir / "table_selective_classification.md").write_text(out)
    print("Wrote table_selective_classification.md")


# Table 4: Hallucination rates per condition per LLM
def table_hallucination(hallu_data: dict, out_dir: Path) -> None:
    """Major / minor / any / mean grounding rate."""
    lines = []
    for llm in LLMS:
        lines.append(f"\n### {llm.title()}\n")
        lines.append("| Condition | Major | Minor | Any | Mean Grounding |")
        lines.append("|---|---|---|---|---|")

        for cond in CONDITIONS_ORDER:
            data = hallu_data.get(llm, {}).get(cond, {})
            stats = data.get("stats", {})
            major = stats.get("major_hallucination_rate")
            minor = stats.get("minor_hallucination_rate")
            any_h = stats.get("any_hallucination_rate")
            ground = stats.get("mean_grounding_rate")
            row = [cond]
            row.append(f"{major:.3f}" if major is not None else "-")
            row.append(f"{minor:.3f}" if minor is not None else "-")
            row.append(f"{any_h:.3f}" if any_h is not None else "-")
            row.append(f"{ground:.3f}" if ground is not None else "-")
            lines.append("| " + " | ".join(row) + " |")

    out = "\n".join(lines)
    (out_dir / "table_hallucination.md").write_text(out)
    print("Wrote table_hallucination.md")


# Figure 1: Confusion matrices for all conditions
def figures_confusion_matrices(eval_data: dict, out_dir: Path) -> None:
    """One PNG per (LLM, condition)."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    for llm in LLMS:
        for cond in CONDITIONS_ORDER:
            data = eval_data.get(llm, {}).get(cond, {})
            cm = data.get("metrics", {}).get("confusion_matrix")
            labels = data.get("metrics", {}).get("labels")
            if cm is None or labels is None:
                continue

            cm = np.array(cm)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Count"}, ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground truth")
            ax.set_title(f"{llm.title()} — {cond}")
            plt.tight_layout()

            safe_cond = cond.replace("+", "_plus_").replace("/", "_")
            out_path = fig_dir / f"cm_{llm}_{safe_cond}.png"
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
    print(f"Wrote {len(LLMS) * len(CONDITIONS_ORDER)} confusion matrix figures (max).")


# Figure 2: Coverage vs Selective Accuracy tradeoff
def figure_selective_tradeoff(eval_data: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"qwen3": "o", "mistral": "s"}
    colors = {"Baseline+Abstain": "#888888",
              "RAG-only+Abstain": "#1f77b4",
              "RAG+NLI+Abstain": "#d62728"}

    for llm in LLMS:
        for cond in ["Baseline+Abstain", "RAG-only+Abstain", "RAG+NLI+Abstain"]:
            sel = eval_data.get(llm, {}).get(cond, {}).get("metrics", {}).get("selective", {})
            cov = sel.get("coverage")
            sa = sel.get("selective_accuracy")
            if cov is None or sa is None:
                continue
            ax.scatter(cov, sa, s=120, marker=markers[llm], color=colors[cond],
                       label=f"{llm} / {cond}", edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective accuracy")
    ax.set_title("Coverage vs Selective Accuracy (abstention-enabled conditions)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "coverage_selective_tradeoff.png", dpi=150)
    plt.close(fig)
    print("Wrote coverage_selective_tradeoff.png")


# Figure 3: Hallucination rates bar chart
def figure_hallucination_bars(hallu_data: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, llm in zip(axes, LLMS):
        conds = []
        major = []
        minor = []
        for cond in CONDITIONS_ORDER:
            data = hallu_data.get(llm, {}).get(cond, {})
            stats = data.get("stats", {})
            if not stats:
                continue
            conds.append(cond)
            major.append(stats.get("major_hallucination_rate", 0))
            minor.append(stats.get("minor_hallucination_rate", 0))

        x = np.arange(len(conds))
        ax.bar(x, major, label="Major", color="#d62728")
        ax.bar(x, minor, bottom=major, label="Minor", color="#ff9896")
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"{llm.title()}")
        ax.set_ylabel("Hallucination rate")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / "hallucination_rates.png", dpi=150)
    plt.close(fig)
    print("Wrote hallucination_rates.png")


# Summary doc consolidating everything
def write_summary(eval_data: dict, hallu_data: dict, out_dir: Path) -> None:
    """One Markdown file with all tables + interpretation hints."""
    lines = [
        f"# Evaluation Summary",
        f"\nGenerated: {datetime.now():%Y-%m-%d %H:%M}",
        f"\n---\n",
        f"## Table 1 — Overall classification performance",
        (out_dir / "table_overall_performance.md").read_text(),
        f"\n---\n",
        f"## Table 2 — Per-class F1 (3-class label set)",
        (out_dir / "table_per_class_f1.md").read_text(),
        f"\n---\n",
        f"## Table 3 — Selective classification (abstention-enabled conditions)",
        (out_dir / "table_selective_classification.md").read_text(),
        f"\n---\n",
        f"## Table 4 — Hallucination rates",
        (out_dir / "table_hallucination.md").read_text(),
        f"\n---\n",
        f"## Quick interpretation checklist",
        "",
        "Compare the following pairs and note any consistent direction across LLMs:",
        "",
        "1. **Baseline vs RAG-only vs RAG+NLI** (without abstain) — does retrieval help?",
        "2. **Each mode with vs without abstain** — does abstention reduce hallucinations?",
        "3. **Coverage vs selective accuracy** — does abstaining on hard cases improve accuracy on remaining cases?",
        "4. **MISLEADING per-class F1** — typically the weakest class; does RAG help or hurt?",
        "5. **Major hallucination rate** — does it drop monotonically with retrieval/abstain?",
        "",
        "If the pattern is consistent across both LLMs, you have a robust finding for the discussion.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))
    print("Wrote summary.md")


# Main
def main(results_dir: str) -> None:
    results = Path(results_dir)
    if not results.is_dir():
        print(f"ERROR: not a directory: {results_dir}")
        sys.exit(1)

    out_dir = results / "thesis_output"
    out_dir.mkdir(exist_ok=True)

    print(f"\nLoading evaluation data from {results}...")
    eval_data = load_eval_data(results)
    hallu_data = load_hallucination_data(results)

    if not eval_data:
        print("\nERROR: no evaluation data loaded. Aborting.")
        sys.exit(1)

    print(f"\nGenerating tables and figures in {out_dir}...")
    table_overall_performance(eval_data, out_dir)
    table_per_class_f1(eval_data, out_dir)
    table_selective(eval_data, out_dir)
    if hallu_data:
        table_hallucination(hallu_data, out_dir)

    figures_confusion_matrices(eval_data, out_dir)
    figure_selective_tradeoff(eval_data, out_dir)
    if hallu_data:
        figure_hallucination_bars(hallu_data, out_dir)

    write_summary(eval_data, hallu_data, out_dir)
    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_thesis_tables.py path/to/results/")
        sys.exit(1)
    main(sys.argv[1])
    