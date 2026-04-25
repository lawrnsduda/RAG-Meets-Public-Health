"""Evaluation: runs claims through all (mode, abstain) conditions."""

from __future__ import annotations
import csv
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from config import (
    PIPELINE_MODES,
    EVAL_LABELS,
    EVAL_LABELS_WITH_ABSTAIN,
    ABSTAIN_LABEL,
    CLAIMS_CSV,
)
from pipeline.retrieval import retrieve_evidence
from pipeline.nli import classify_nli
from pipeline.verdict import generate_verdict

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 6 conditions = 3 modes x 2 abstain settings.
ALL_CONDITIONS: list[tuple[str, bool]] = [
    (mode, abstain)
    for mode in PIPELINE_MODES.keys()
    for abstain in (False, True)
]


def condition_name(mode: str, use_abstain: bool) -> str:
    return f"{mode}+Abstain" if use_abstain else mode


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
    use_abstain: bool = False,
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
        use_abstain=use_abstain,
    )

    return {
        "verdict": result["verdict"],
        "justification": result["justification"],
        "evidence_count": len(evidence),
        "nli_labels": [r["label"] for r in nli_results] if nli_results else [],
    }


def compute_selective_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Coverage, selective accuracy, abstention rate per class."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    covered = y_pred_arr != ABSTAIN_LABEL
    coverage = float(covered.mean()) if len(covered) else 0.0

    if covered.sum() == 0:
        selective_acc = float("nan")
    else:
        correct = (y_true_arr[covered] == y_pred_arr[covered]).sum()
        selective_acc = float(correct) / int(covered.sum())

    abstention_by_class = {}
    for cls in EVAL_LABELS:
        mask = y_true_arr == cls
        if mask.sum() > 0:
            abstention_by_class[cls] = float(
                (y_pred_arr[mask] == ABSTAIN_LABEL).mean()
            )
        else:
            abstention_by_class[cls] = float("nan")

    return {
        "coverage": round(coverage, 4),
        "selective_accuracy": (
            round(selective_acc, 4) if not np.isnan(selective_acc) else None
        ),
        "overall_abstention_rate": round(1.0 - coverage, 4),
        "abstention_by_class": {
            k: round(v, 4) if not np.isnan(v) else None
            for k, v in abstention_by_class.items()
        },
    }


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
    use_abstain: bool = False,
) -> dict:
    labels = EVAL_LABELS_WITH_ABSTAIN if use_abstain else EVAL_LABELS

    try:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            labels=labels, output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "per_class": {
                label: {
                    "precision": round(report[label]["precision"], 4),
                    "recall": round(report[label]["recall"], 4),
                    "f1": round(report[label]["f1-score"], 4),
                    "support": report[label]["support"],
                }
                for label in labels if label in report
            },
            "confusion_matrix": cm.tolist(),
            "labels": labels,
        }

    except ImportError:
        print("  ⚠ scikit-learn not found — computing basic accuracy only.")
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return {
            "accuracy": round(correct / len(y_true), 4) if y_true else 0,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": [],
            "labels": labels,
        }


def run_evaluation(
    llm_choice: str = "qwen3",
    conditions: list[tuple[str, bool]] | None = None,
):
    if conditions is None:
        conditions = ALL_CONDITIONS

    claims = load_claims()
    print(f"Loaded {len(claims)} claims from {CLAIMS_CSV}")

    all_results = {}

    for mode, use_abstain in conditions:
        cond_label = condition_name(mode, use_abstain)
        print(f"\n{'='*60}")
        print(f"  Condition: {cond_label}")
        print(f"{'='*60}")

        y_true = []
        y_pred = []
        details = []

        for i, claim in enumerate(claims):
            print(f"  [{i+1}/{len(claims)}] {claim['Claim'][:60]}…")

            result = run_single_claim(
                claim_text=claim["Claim"],
                mode=mode,
                llm_choice=llm_choice,
                use_abstain=use_abstain,
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

        metrics = compute_metrics(y_true, y_pred, use_abstain=use_abstain)

        if use_abstain:
            metrics["selective"] = compute_selective_metrics(y_true, y_pred)

        all_results[cond_label] = {
            "mode": mode,
            "use_abstain": use_abstain,
            "metrics": metrics,
            "details": details,
        }

        print(f"\n  Accuracy: {metrics['accuracy']}")
        print(f"  Macro F1: {metrics['macro_f1']}")
        if use_abstain:
            sel = metrics["selective"]
            print(f"  Coverage: {sel['coverage']}")
            print(f"  Selective accuracy: {sel['selective_accuracy']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{llm_choice}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return all_results


if __name__ == "__main__":
    for llm in ["qwen3", "mistral"]:
        print(f"\n\n{'#'*60}\n#  LLM: {llm}\n{'#'*60}")
        run_evaluation(llm_choice=llm)