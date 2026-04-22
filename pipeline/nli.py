"""
Natural Language Inference using cross-encoder/nli-deberta-v3-small.

This module checks for each (claim, evidence) pair whether the evidence
supports, contradicts, or is neutral towards the claim. We use a
cross-encoder here because it processes both texts together through the
transformer, which gives a lot better results than encoding them separately
(bi-encoder). It's slower, but we only have a handful of passages per
claim so that should be alright.
"""
from __future__ import annotations
from config import NLI_MODEL, NLI_LABELS

_classifier = None


def _load_model():

    """Load tokenizer + model from HuggingFace.
    Only happens once, then it's cached in the global _classifier variable.
    We call model.eval() so things like dropout are turned off during inference."""
 
    global _classifier
    if _classifier is None:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            NLI_MODEL
        )
        model.eval() 
        _classifier = (tokenizer, model)
    return _classifier


def classify_nli(
    claim: str,
    evidence_passages: list[dict],
) -> list[dict]:
    
    """
    Classify each (claim, passage) pair.
    Returns list of dicts with keys: label, scores
    """
    
    import torch

    tokenizer, model = _load_model()
    results: list[dict] = []

    for passage in evidence_passages:
        encoded = tokenizer(
            claim,
            passage["text"],
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits 
            probs = torch.softmax(logits, dim=-1).squeeze().tolist() # convert to probabilities

        score_map = {
            NLI_LABELS[i]: round(probs[i], 4) for i in range(3)
        }
        predicted_label = max(score_map, key=score_map.get)

        results.append({
            "label": predicted_label,
            "scores": score_map,
        })

    return results