# Appendix C — Generation Parameters

This appendix records the models and parameters I used in the evaluation
pipeline. The same values are defined in [`../config.py`](../config.py) and the
evaluation scripts — the tables below are the human-readable summary.

## C.1 Language models (verdict generation)

Both LLMs are served **locally via Ollama** — no cloud LLM is used in the
pipeline.

| Role | Model | Ollama model ID | Size |
|---|---|---|---|
| Verdict LLM | Qwen3 8B | `qwen3:8b` (`500a1f067a9f`) | 5.2 GB |
| Verdict LLM | Mistral 7B | `mistral:7b` (`6577803aa9a0`) | 4.4 GB |

Both models use **Q4_K_M** GGUF quantisation and are served with **Ollama 0.30.10**.

## C.2 Decoding parameters

| Parameter | Value |
|---|---|
| Temperature | 0.1 |
| Max new tokens (`num_predict`) | 1024 |
| Labels (ground truth) | `SUPPORTED`, `REFUTED`, `MISLEADING` |
| Abstain label (prediction-only) | `NOT_ENOUGH_EVIDENCE` |

## C.3 Retrieval

| Parameter | Value |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (384-dimensional vectors) |
| Vector store | FAISS |
| Chunk size | 200 words |
| Chunk overlap | 50 words |
| Top-k retrieved | 5 |
| Evidence corpus | 723 passages (WHO 500, PubMed/NCBI 169, CDC 38, ECDC 16) |

> The OWID source is enabled in `config.py` but yields no usable passages; the
> evaluated index is built from the four working sources (WHO, PubMed, CDC, ECDC).

## C.4 NLI stage

| Parameter | Value |
|---|---|
| NLI model | `cross-encoder/nli-deberta-v3-small` (DeBERTa-v3-small) |
| NLI threshold τ | 0.6 |
| NLI classes | contradiction / entailment / neutral |

## C.5 Experimental conditions

Six conditions per LLM (3 retrieval modes × 2 abstain settings):

- Baseline, Baseline+Abstain
- RAG-only, RAG-only+Abstain
- RAG+NLI, RAG+NLI+Abstain
