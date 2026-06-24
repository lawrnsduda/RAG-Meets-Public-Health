# Appendix D — Runtime and Hardware

This appendix records the environment in which the evaluation was run.

## D.1 Evaluation environment

| Item | Value |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA | 13.0 |
| LLM serving | Ollama 0.30.10 (local, no cloud LLM) |
| Python | 3.11.15 |
| torch | 2.2.2 |
| torchvision | 0.17.2 |
| transformers | 4.57.6 |
| sentence-transformers | 3.4.1 |
| faiss-cpu | 1.13.2 |

## D.2 Run timing

The raw per-run outputs are committed under
[`../evaluation/results/`](../evaluation/results/) (the `eval_*.json` files), each
named with the model and run timestamp. The two final runs used for the thesis
results are:

- `eval_mistral_20260427_120729.json`
- `eval_qwen3_20260427_115110.json`

> **EINFÜGEN:** per-condition wall-clock timing (total evaluation time and, where
> available, mean seconds per claim) for each model and condition. Read these from
> the run logs / timing fields on the local machine — the committed JSONs store
> predictions and metrics but not wall-clock timing.

## D.3 Reproducing the runs

```bash
python -m evaluation.evaluate                 # all 6 conditions per LLM -> results/*.json
python -m evaluation.hallucination_analysis   # per-sentence hallucination check
python -m evaluation.build_thesis_tables      # regenerate thesis_output/ tables + figures
```
