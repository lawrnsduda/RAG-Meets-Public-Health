"""
Evaluation: runs claims through 3 conditions, computes metrics.
"""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from config import PIPELINE_MODES, EVAL_LABELS, CLAIMS_CSV
from pipeline.retrieval import retrieve_evidence
from pipeline.nli import classify_nli
from pipeline.verdict import generate_verdict

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_claims() -> list[dict]:
    claims = []
    with open(CLAIMS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claims.append(row)
    return claims


def run_single_claim(
    claim_text: str,
    mode: str,
    llm_choice: str = "qwen3",
    top_k: int = 5,
) -> dict:
    mode_cfg = PIPELINE_MODES[mode]

    evidence = []
    nli_results = []

    if mode_cfg["use_retrieval"]:
        evidence = retrieve_evidence(claim_text, top_k=top_k)

    if mode_cfg["use_nli"] and evidence:
        nli_results = classify_nli(claim_text, evidence)

    result = generate_verdict(
        claim=claim_text,
        evidence=evidence,
        nli_labels=nli_results,
        mode=mode,
        llm_choice=llm_choice,
    )

    return {
        "verdict": result["verdict"],
        "justification": result["justification"],
        "evidence_count": len(evidence),
        "nli_labels": [r["label"] for r in nli_results]
        if nli_results
        else [],
    }


def compute_metrics(
    y_true: list[str], y_pred: list[str]
) -> dict:
    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            labels=EVAL_LABELS,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(
            y_true, y_pred, labels=EVAL_LABELS
        )

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(
                report["macro avg"]["f1-score"], 4
            ),
            "per_class": {
                label: {
                    "precision": round(
                        report[label]["precision"], 4
                    ),
                    "recall": round(
                        report[label]["recall"], 4
                    ),
                    "f1": round(
                        report[label]["f1-score"], 4
                    ),
                    "support": report[label]["support"],
                }
                for label in EVAL_LABELS
                if label in report
            },
            "confusion_matrix": cm.tolist(),
        }

    except ImportError:
        print(
            "  ⚠ scikit-learn not found — "
            "computing basic accuracy only."
        )
        correct = sum(
            1 for t, p in zip(y_true, y_pred) if t == p
        )
        return {
            "accuracy": round(correct / len(y_true), 4)
            if y_true
            else 0,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": [],
        }


def run_evaluation(
    llm_choice: str = "qwen3",
    modes: list[str] | None = None,
):
    if modes is None:
        modes = list(PIPELINE_MODES.keys())

    claims = load_claims()
    print(f"Loaded {len(claims)} claims from {CLAIMS_CSV}")

    all_results = {}

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Condition: {mode}")
        print(f"{'='*60}")

        y_true = []
        y_pred = []
        details = []

        for i, claim in enumerate(claims):
            print(
                f"  [{i+1}/{len(claims)}] "
                f"{claim['Claim'][:60]}…"
            )

            result = run_single_claim(
                claim_text=claim["Claim"],
                mode=mode,
                llm_choice=llm_choice,
            )

            y_true.append(claim["Label"])
            y_pred.append(result["verdict"])
            details.append({
                "id": claim["ID"],
                "claim": claim["Claim"],
                "ground_truth": claim["Label"],
                "predicted": result["verdict"],
                "justification": result["justification"],
                "evidence_count": result["evidence_count"],
                "nli_labels": result["nli_labels"],
            })

            time.sleep(0.1)

        metrics = compute_metrics(y_true, y_pred)
        all_results[mode] = {
            "metrics": metrics,
            "details": details,
        }

        print(f"\n  Accuracy: {metrics['accuracy']}")
        print(f"  Macro F1: {metrics['macro_f1']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    for llm in ["qwen3", "mistral"]:
        print(f"\n\n{'#'*60}\n#  LLM: {llm}\n{'#'*60}")
        run_evaluation(llm_choice=llm)