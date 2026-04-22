"""
fact_checker – LLM-based verdict generation.

This is the last step of the pipeline. We take the claim, the retrieved
evidence, and the NLI labels, put it all into a prompt, send it to a
local Ollama model (Qwen3 8B or Mistral 7B), and parse the response
into a structured verdict (SUPPORTED / REFUTED / MISLEADING).
"""

from __future__ import annotations
from config import (
    OLLAMA_MODEL,
    OLLAMA_MISTRAL_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EVAL_LABELS,
)

SYSTEM_PROMPT = (
    "You are a fact_checker assistant. Your task is to classify a "
    "health claim as SUPPORTED, REFUTED, or MISLEADING and provide a "
    "short justification. Base your verdict strictly on the provided "
    "evidence. If no evidence is provided, use your general knowledge "
    "but flag uncertainty."
)


def _build_prompt(
    claim: str,
    evidence: list[dict],
    nli_labels: list[dict],
    mode: str,
) -> str:
    
    """
    Build the actual prompt that gets sent to the LLM.
    The structure changes depending on the pipeline mode:
 
    - Baseline:  no evidence at all, model has to use its own knowledge
    - RAG-only:  evidence is listed but without NLI labels
    - RAG+NLI:   evidence is listed WITH the NLI label for each passage
                  so the LLM can see right away if a passage supports
                  or contradicts the claim
 
    At the end we always add a strict format instruction so we can
    parse the response afterwards.
    """

    parts = [f"CLAIM: {claim}\n"]

    if mode == "Baseline":
        parts.append(
            "No external evidence is provided. Classify based on your "
            "parametric knowledge. Flag any uncertainty.\n"
        )
    elif mode == "RAG-only":
        parts.append("RETRIEVED EVIDENCE:\n")
        for i, ev in enumerate(evidence, 1):
            parts.append(
                f"  [{i}] ({ev['source']}) {ev['text']}\n"
            )
    elif mode == "RAG+NLI":
        parts.append("RETRIEVED EVIDENCE WITH NLI LABELS:\n")
        for i, (ev, nli) in enumerate(
            zip(evidence, nli_labels), 1
        ):
            parts.append(
                f"  [{i}] ({ev['source']}) [{nli['label'].upper()}] "
                f"{ev['text']}\n"
            )

    parts.append(
        "\nRespond with exactly:\n"
        "VERDICT: <SUPPORTED | REFUTED | MISLEADING>\n"
        "JUSTIFICATION: <2-4 sentences referencing the evidence>\n"
    )

    return "".join(parts)


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:

    """
    Send the prompt to Ollama and return the raw text response.
    Uses a low temperature so the output is fairly deterministic,
    and caps the token count so it doesn't "go on" forever.
    """

    import ollama

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": LLM_TEMPERATURE,
            "num_predict": LLM_MAX_TOKENS,
        },
    )
    return response["message"]["content"]


def _parse_response(raw: str) -> dict:

    """
    Try to extract verdict + justification from the LLM output.
 
    We go through the response line by line and look for lines
    starting with "VERDICT:" and "JUSTIFICATION:".
    If the model didn't follow the format properly, verdict defaults
    to "UNKNOWN" and we just return the whole raw output as justification
    so the user can still see what happened.
    """

    verdict = "UNKNOWN"
    justification = raw.strip() 
    for line in raw.splitlines():
        line_upper = line.strip().upper()
        if line_upper.startswith("VERDICT:"):
            token = line_upper.replace("VERDICT:", "").strip()
            for label in EVAL_LABELS:
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
    
    """
    Main function of this module – ties everything together.
 
    1. Build the prompt based on which mode we're in
    2. Send it to the right Ollama model (qwen3 or mistral)
    3. Parse the response and return a clean dict
 
    Returns a dict with: verdict, justification, raw
    """

    prompt = _build_prompt(claim, evidence, nli_labels, mode)

    if llm_choice == "mistral":
        raw = _call_ollama(prompt, model=OLLAMA_MISTRAL_MODEL)
    else:
        raw = _call_ollama(prompt, model=OLLAMA_MODEL)

    return _parse_response(raw)

