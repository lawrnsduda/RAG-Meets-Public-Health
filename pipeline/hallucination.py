"""
Automatic hallucination detection via NLI.

For each sentence in a generated justification, we run the same DeBERTa
NLI model used in the RAG+NLI pipeline against the claim + evidence
passages and classify it as contradicted (major), entailed (supported),
or neither (minor/unsupported). Per-justification these aggregate into
hallucination-free / minor / major, matching Section 5.5.1 of the thesis.
A stratified subsample should still be manually validated; Baseline has
no evidence, so the automatic check is only weakly meaningful there.
"""

from __future__ import annotations
import re

from pipeline.nli import classify_nli


MIN_SENTENCE_LENGTH = 15

NLI_CONFIDENCE_THRESHOLD = 0.6


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_LENGTH]


def _check_support_and_contradiction(
    sentence: str,
    grounding_premises: list[dict],
    contradiction_premises: list[dict],
) -> dict:
    if grounding_premises:
        entail_results = classify_nli(sentence, grounding_premises)
        max_entail = max(
            (r["scores"].get("entailment", 0) for r in entail_results),
            default=0.0,
        )
        is_supported = max_entail >= NLI_CONFIDENCE_THRESHOLD
    else:
        max_entail = 0.0
        is_supported = False

    if contradiction_premises:
        contra_results = classify_nli(sentence, contradiction_premises)
        max_contra = max(
            (r["scores"].get("contradiction", 0) for r in contra_results),
            default=0.0,
        )
        is_contradicted = max_contra >= NLI_CONFIDENCE_THRESHOLD
    else:
        max_contra = 0.0
        is_contradicted = False

    if is_contradicted:
        status = "contradicted"
    elif is_supported:
        status = "supported"
    else:
        status = "unsupported"

    return {
        "sentence": sentence,
        "status": status,
        "max_entailment": round(max_entail, 4),
        "max_contradiction": round(max_contra, 4),
    }


def check_hallucination(
    justification: str,
    claim: str,
    evidence: list[dict],
) -> dict:
    sentences = split_into_sentences(justification)
    if not sentences:
        return {
            "category": "unclassifiable",
            "n_sentences": 0,
            "n_supported": 0,
            "n_contradicted": 0,
            "n_unsupported": 0,
            "grounding_rate": None,
            "sentences": [],
        }

    grounding_premises: list[dict] = [
        {"text": claim, "source": "claim"}
    ]
    for ev in evidence:
        grounding_premises.append(
            {"text": ev["text"], "source": ev.get("source", "?")}
        )

    contradiction_premises: list[dict] = [
        {"text": ev["text"], "source": ev.get("source", "?")}
        for ev in evidence
    ]

    analyses = [
        _check_support_and_contradiction(
            s, grounding_premises, contradiction_premises
        )
        for s in sentences
    ]

    counts = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for a in analyses:
        counts[a["status"]] += 1

    n = len(analyses)
    if counts["contradicted"] > 0:
        category = "major_hallucination"
    elif counts["unsupported"] > 0:
        category = "minor_hallucination"
    else:
        category = "hallucination_free"

    return {
        "category": category,
        "n_sentences": n,
        "n_supported": counts["supported"],
        "n_contradicted": counts["contradicted"],
        "n_unsupported": counts["unsupported"],
        "grounding_rate": round(counts["supported"] / n, 4) if n else None,
        "sentences": analyses,
    }


def aggregate_hallucination_stats(
    per_claim_results: list[dict],
) -> dict:
    if not per_claim_results:
        return {}

    classifiable = [
        r for r in per_claim_results if r["category"] != "unclassifiable"
    ]
    total = len(classifiable)
    if total == 0:
        return {"n_total": 0}

    major = sum(1 for r in classifiable if r["category"] == "major_hallucination")
    minor = sum(1 for r in classifiable if r["category"] == "minor_hallucination")
    clean = sum(1 for r in classifiable if r["category"] == "hallucination_free")

    grounding_rates = [
        r["grounding_rate"] for r in classifiable
        if r["grounding_rate"] is not None
    ]
    mean_grounding = (
        round(sum(grounding_rates) / len(grounding_rates), 4)
        if grounding_rates else None
    )

    return {
        "n_total": total,
        "n_unclassifiable": len(per_claim_results) - total,
        "major_hallucination_rate": round(major / total, 4),
        "minor_hallucination_rate": round(minor / total, 4),
        "any_hallucination_rate": round((major + minor) / total, 4),
        "hallucination_free_rate": round(clean / total, 4),
        "mean_grounding_rate": mean_grounding,
    }


def stratified_sample(
    per_claim_results: list[dict],
    details: list[dict],
    n_per_category: int = 7,
    seed: int = 42,
) -> list[dict]:
    import random
    rng = random.Random(seed)

    assert len(per_claim_results) == len(details), (
        "per_claim_results and details must have same length"
    )

    buckets: dict[str, list[int]] = {
        "major_hallucination": [],
        "minor_hallucination": [],
        "hallucination_free": [],
    }
    for idx, r in enumerate(per_claim_results):
        cat = r["category"]
        if cat in buckets:
            buckets[cat].append(idx)

    sampled_indices: list[int] = []
    for cat, idxs in buckets.items():
        k = min(n_per_category, len(idxs))
        sampled_indices.extend(rng.sample(idxs, k))

    rows = []
    for idx in sampled_indices:
        d = details[idx]
        h = per_claim_results[idx]
        rows.append({
            "id": d.get("id", ""),
            "claim": d.get("claim", ""),
            "ground_truth": d.get("ground_truth", ""),
            "predicted": d.get("predicted", ""),
            "justification": d.get("justification", ""),
            "auto_category": h["category"],
            "auto_grounding_rate": h["grounding_rate"],
            # To be filled in manually:
            "manual_category": "",  # hallucination_free | minor_hallucination | major_hallucination
            "annotator_notes": "",
        })
    return rows