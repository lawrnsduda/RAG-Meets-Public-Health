"""
Post-hoc hallucination analysis.

Reads eval_*.json files, re-retrieves evidence for each claim, runs
the per-sentence NLI hallucination check, and writes aggregated stats
plus stratified samples for manual validation.

Usage:
    python -m evaluation.hallucination_analysis path/to/eval.json
    python -m evaluation.hallucination_analysis   # processes all eval JSONs
"""

from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from datetime import datetime

from config import PIPELINE_MODES
from pipeline.retrieval import retrieve_evidence
from pipeline.hallucination import (
    check_hallucination,
    aggregate_hallucination_stats,
    stratified_sample,
)

RESULTS_DIR = Path(__file__).parent / "results"


def analyse_run(eval_json_path: Path) -> dict:
    with open(eval_json_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    output = {}

    for condition, cond_data in eval_data.items():
        details = cond_data.get("details", [])
        if not details:
            continue

        # Determine retrieval status from the 'mode' field; fall back
        # to checking the condition name for older eval JSONs.
        mode = cond_data.get("mode")
        if mode and mode in PIPELINE_MODES:
            uses_retrieval = PIPELINE_MODES[mode]["use_retrieval"]
        else:
            uses_retrieval = "RAG" in condition

        print(f"\n--- Condition: {condition} ({len(details)} claims) ---")

        per_claim = []
        for i, d in enumerate(details):
            claim = d["claim"]
            justification = d.get("justification", "")

            evidence = retrieve_evidence(claim, top_k=5) if uses_retrieval else []

            result = check_hallucination(justification, claim, evidence)
            per_claim.append(result)

            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(details)}] processed")

        stats = aggregate_hallucination_stats(per_claim)
        output[condition] = {
            "stats": stats,
            "per_claim": per_claim,
            "details": details,
        }

        print(f"  Major hallucination rate: {stats['major_hallucination_rate']}")
        print(f"  Minor hallucination rate: {stats['minor_hallucination_rate']}")
        print(f"  Any hallucination rate:   {stats['any_hallucination_rate']}")
        print(f"  Mean grounding rate:      {stats['mean_grounding_rate']}")

    return output


def write_summary_table(results: dict, llm_name: str) -> None:
    print(f"\n{'='*80}")
    print(f"  Hallucination summary - {llm_name}")
    print(f"{'='*80}")
    print(f"{'Condition':<25} {'Major':>8} {'Minor':>8} {'Any':>8} {'Ground':>8}")
    print("-" * 80)
    for condition, data in results.items():
        s = data["stats"]
        print(
            f"{condition:<25} "
            f"{s['major_hallucination_rate']:>8.2%} "
            f"{s['minor_hallucination_rate']:>8.2%} "
            f"{s['any_hallucination_rate']:>8.2%} "
            f"{s['mean_grounding_rate']:>8.2%}"
        )


def write_sample_csv(
    results: dict,
    llm_name: str,
    timestamp: str,
    n_per_category: int = 7,
) -> None:
    """Stratified samples for manual validation. ~21 items per condition."""
    sample_dir = RESULTS_DIR / f"hallucination_samples_{llm_name}_{timestamp}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for condition, data in results.items():
        rows = stratified_sample(
            per_claim_results=data["per_claim"],
            details=data["details"],
            n_per_category=n_per_category,
        )
        if not rows:
            continue

        safe_name = condition.replace("+", "_plus_").replace("/", "_")
        out_path = sample_dir / f"sample_{safe_name}.csv"

        fieldnames = list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote sample: {out_path} ({len(rows)} items)")


def main(arg_path: str | None = None) -> None:
    if arg_path:
        paths = [Path(arg_path)]
    else:
        paths = sorted(RESULTS_DIR.glob("eval_*.json"))

    if not paths:
        print(f"No eval JSONs found in {RESULTS_DIR}")
        return

    for path in paths:
        print(f"\n\nProcessing {path.name}...")
        parts = path.stem.split("_")
        llm_name = parts[1] if len(parts) >= 2 else "unknown"

        results = analyse_run(path)
        if not results:
            continue

        write_summary_table(results, llm_name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        out_json = RESULTS_DIR / f"hallucination_{llm_name}_{timestamp}.json"
        writeable = {
            cond: {"stats": d["stats"], "per_claim": d["per_claim"]}
            for cond, d in results.items()
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(writeable, f, indent=2, ensure_ascii=False)
        print(f"\nFull results: {out_json}")

        write_sample_csv(results, llm_name, timestamp)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)