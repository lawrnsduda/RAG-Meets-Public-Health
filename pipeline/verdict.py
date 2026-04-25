"""
fact_checker – LLM-based verdict generation.

Final step of the pipeline: build a prompt from the claim + evidence
(+ optional NLI labels), send it to a local Ollama model (Qwen3 8B or
Mistral 7B), and parse the response.

Standard modes predict SUPPORTED / REFUTED / MISLEADING. Abstain modes
add a fourth NOT_ENOUGH_EVIDENCE label to address the over-refutation
bias we saw in the first evaluation — where missing evidence was being
read as REFUTED instead of "I don't know".
"""

from __future__ import annotations
from config import (
    OLLAMA_MODEL,
    OLLAMA_MISTRAL_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EVAL_LABELS,
    EVAL_LABELS_WITH_ABSTAIN,
    PIPELINE_MODES,
)

SYSTEM_PROMPT = (
    "You are a fact_checker assistant. Your task is to classify a "
    "health claim as SUPPORTED, REFUTED, or MISLEADING and provide a "
    "short justification. Base your verdict strictly on the provided "
    "evidence. If no evidence is provided, use your general knowledge "
    "but flag uncertainty."
)

# Explicit "do not default to REFUTED" rule is what addresses the
# over-refutation bias; without it the 4-class labels alone are not enough.
SYSTEM_PROMPT_ABSTAIN = (
    "You are a fact_checker assistant. Your task is to classify a "
    "health claim into exactly ONE of four categories:\n"
    "  - SUPPORTED: evidence explicitly confirms the claim.\n"
    "  - REFUTED: evidence explicitly contradicts the claim.\n"
    "  - MISLEADING: the claim is partially true but cherry-picked, "
    "oversimplified, or missing critical context.\n"
    "  - NOT_ENOUGH_EVIDENCE: the retrieved evidence is insufficient to "
    "confirm, refute, or identify the claim as misleading.\n\n"
    "IMPORTANT RULES:\n"
    "  1. Do NOT default to REFUTED when the evidence does not directly "
    "address the claim. REFUTED requires an explicit contradiction.\n"
    "  2. Use NOT_ENOUGH_EVIDENCE when the passages are absent, generic, "
    "or tangential to the claim.\n"
    "  3. Base your verdict strictly on the provided evidence."
)


def _build_prompt(
    claim: str,
    evidence: list[dict],
    nli_labels: list[dict],
    mode: str,
) -> str:
    parts = [f"CLAIM: {claim}\n"]
    mode_cfg = PIPELINE_MODES.get(mode, {})
    use_retrieval = mode_cfg.get("use_retrieval", False)
    use_nli = mode_cfg.get("use_nli", False)
    use_abstain = mode_cfg.get("use_abstain", False)

    if not use_retrieval:
        parts.append(
            "No external evidence is provided. Classify based on your "
            "parametric knowledge. Flag any uncertainty.\n"
        )
    elif use_nli:
        parts.append("RETRIEVED EVIDENCE WITH NLI LABELS:\n")
        for i, (ev, nli) in enumerate(zip(evidence, nli_labels), 1):
            parts.append(
                f"  [{i}] ({ev['source']}) [{nli['label'].upper()}] "
                f"{ev['text']}\n"
            )
    else:
        parts.append("RETRIEVED EVIDENCE:\n")
        for i, ev in enumerate(evidence, 1):
            parts.append(
                f"  [{i}] ({ev['source']}) {ev['text']}\n"
            )

    if use_abstain:
        parts.append(
            "\nRespond with exactly:\n"
            "VERDICT: <SUPPORTED | REFUTED | MISLEADING | NOT_ENOUGH_EVIDENCE>\n"
            "JUSTIFICATION: <2-4 sentences referencing the evidence>\n"
        )
    else:
        parts.append(
            "\nRespond with exactly:\n"
            "VERDICT: <SUPPORTED | REFUTED | MISLEADING>\n"
            "JUSTIFICATION: <2-4 sentences referencing the evidence>\n"
        )

    return "".join(parts)


def _call_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


def _parse_response(raw: str, mode: str) -> dict:
    # If the model's format breaks, fall back to verdict=UNKNOWN and
    # return the full raw text as the justification so users can still
    # see what went wrong.
    mode_cfg = PIPELINE_MODES.get(mode, {})
    use_abstain = mode_cfg.get("use_abstain", False)
    valid_labels = EVAL_LABELS_WITH_ABSTAIN if use_abstain else EVAL_LABELS

    verdict = "UNKNOWN"
    justification = raw.strip()
    for line in raw.splitlines():
        line_upper = line.strip().upper()
        if line_upper.startswith("VERDICT:"):
            # Normalize "NOT ENOUGH EVIDENCE" -> "NOT_ENOUGH_EVIDENCE"
            # so the model's whitespace style doesn't break parsing.
            token = line_upper.replace("VERDICT:", "").strip()
            token = token.replace(" ", "_")
            for label in valid_labels:
                if label in token:
                    verdict = label
                    break
        elif line.strip().upper().startswith("JUSTIFICATION:"):
            justification = (
                line.strip()[len("JUSTIFICATION:"):].strip()
            )

    return {
        "verdict": verdict,
        "justification": justification,
        "raw": raw,
    }


def generate_verdict(
    claim: str,
    evidence: list[dict],
    nli_labels: list[dict],
    mode: str,
    llm_choice: str = "qwen3",
) -> dict:
    prompt = _build_prompt(claim, evidence, nli_labels, mode)

    mode_cfg = PIPELINE_MODES.get(mode, {})
    system_prompt = (
        SYSTEM_PROMPT_ABSTAIN if mode_cfg.get("use_abstain") else SYSTEM_PROMPT
    )

    model_name = (
        OLLAMA_MISTRAL_MODEL if llm_choice == "mistral" else OLLAMA_MODEL
    )

    raw = _call_ollama(prompt, model=model_name, system_prompt=system_prompt)

    return _parse_response(raw, mode)
