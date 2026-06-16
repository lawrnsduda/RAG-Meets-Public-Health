# Evaluation Summary

Generated: 2026-05-03 15:13

---

## Table 1 — Overall classification performance
| Condition | Qwen3 Acc | Qwen3 F1 | Mistral Acc | Mistral F1 |
|---|---|---|---|---|
| Baseline | 0.513 | 0.517 | 0.613 | 0.626 |
| Baseline+Abstain | 0.667 | 0.529 | 0.640 | 0.507 |
| RAG-only | 0.453 | 0.414 | 0.560 | 0.536 |
| RAG-only+Abstain | 0.367 | 0.350 | 0.573 | 0.469 |
| RAG+NLI | 0.447 | 0.442 | 0.553 | 0.528 |
| RAG+NLI+Abstain | 0.313 | 0.315 | 0.533 | 0.449 |

---

## Table 2 — Per-class F1 (3-class label set)

### Qwen3

| Condition | F1-SUPPORTED | F1-REFUTED | F1-MISLEADING |
|---|---|---|---|
| Baseline | 0.718 | 0.533 | 0.299 |
| Baseline+Abstain | 0.718 | 0.717 | 0.680 |
| RAG-only | 0.543 | 0.570 | 0.131 |
| RAG-only+Abstain | 0.438 | 0.538 | 0.427 |
| RAG+NLI | 0.522 | 0.578 | 0.225 |
| RAG+NLI+Abstain | 0.462 | 0.444 | 0.353 |

### Mistral

| Condition | F1-SUPPORTED | F1-REFUTED | F1-MISLEADING |
|---|---|---|---|
| Baseline | 0.829 | 0.554 | 0.495 |
| Baseline+Abstain | 0.860 | 0.588 | 0.581 |
| RAG-only | 0.738 | 0.542 | 0.329 |
| RAG-only+Abstain | 0.760 | 0.451 | 0.667 |
| RAG+NLI | 0.722 | 0.603 | 0.257 |
| RAG+NLI+Abstain | 0.753 | 0.442 | 0.602 |

---

## Table 3 — Selective classification (abstention-enabled conditions)

### Qwen3

| Condition | Coverage | Selective Acc | Abst-SUP | Abst-REF | Abst-MIS |
|---|---|---|---|---|---|
| Baseline+Abstain | 0.913 | 0.730 | 0.180 | 0.060 | 0.020 |
| RAG-only+Abstain | 0.600 | 0.611 | 0.520 | 0.360 | 0.320 |
| RAG+NLI+Abstain | 0.500 | 0.627 | 0.600 | 0.540 | 0.360 |

### Mistral

| Condition | Coverage | Selective Acc | Abst-SUP | Abst-REF | Abst-MIS |
|---|---|---|---|---|---|
| Baseline+Abstain | 0.853 | 0.750 | 0.080 | 0.220 | 0.140 |
| RAG-only+Abstain | 0.780 | 0.735 | 0.240 | 0.300 | 0.120 |
| RAG+NLI+Abstain | 0.753 | 0.708 | 0.240 | 0.340 | 0.160 |

---

## Table 4 — Hallucination rates

### Qwen3

| Condition | Major | Minor | Any | Mean Grounding |
|---|---|---|---|---|
| Baseline | 0.000 | 1.000 | 1.000 | 0.062 |
| Baseline+Abstain | 0.000 | 1.000 | 1.000 | 0.063 |
| RAG-only | 0.367 | 0.626 | 0.993 | 0.089 |
| RAG-only+Abstain | 0.380 | 0.613 | 0.993 | 0.087 |
| RAG+NLI | 0.409 | 0.583 | 0.992 | 0.086 |
| RAG+NLI+Abstain | 0.392 | 0.601 | 0.993 | 0.091 |

### Mistral

| Condition | Major | Minor | Any | Mean Grounding |
|---|---|---|---|---|
| Baseline | 0.000 | 0.993 | 0.993 | 0.103 |
| Baseline+Abstain | 0.000 | 0.993 | 0.993 | 0.118 |
| RAG-only | 0.340 | 0.640 | 0.980 | 0.150 |
| RAG-only+Abstain | 0.287 | 0.680 | 0.967 | 0.191 |
| RAG+NLI | 0.347 | 0.613 | 0.960 | 0.151 |
| RAG+NLI+Abstain | 0.253 | 0.687 | 0.940 | 0.191 |

---

## Quick interpretation checklist

Compare the following pairs and note any consistent direction across LLMs:

1. **Baseline vs RAG-only vs RAG+NLI** (without abstain) — does retrieval help?
2. **Each mode with vs without abstain** — does abstention reduce hallucinations?
3. **Coverage vs selective accuracy** — does abstaining on hard cases improve accuracy on remaining cases?
4. **MISLEADING per-class F1** — typically the weakest class; does RAG help or hurt?
5. **Major hallucination rate** — does it drop monotonically with retrieval/abstain?

If the pattern is consistent across both LLMs, you have a robust finding for the discussion.