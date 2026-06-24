# Appendix — RAG Meets Public Health

This directory is the **reproducibility appendix** of my master's thesis
*"RAG Meets Public Health: Development and Evaluation of a Hybrid System for
Automated Fact-Checking"* (Laurence Douda, FH St. Pölten).

I moved this material out of the thesis document and collected it here so that all
artefacts — the evaluation dataset, the verbatim prompts, the generation
parameters, and the runtime / hardware setup — sit in one place, next to the code
they describe.

## Contents

| File | Appendix | Content |
|---|---|---|
| [`A_dataset.md`](A_dataset.md) | A | Measles claim dataset: schema, label distribution, ID mapping, construction and verification procedure |
| [`B_prompts.md`](B_prompts.md) | B | Verbatim prompt templates (dataset generation + pipeline verdict prompts) |
| [`C_generation_parameters.md`](C_generation_parameters.md) | C | Models, decoding parameters, retrieval/NLI hyperparameters |
| [`D_runtime_and_hardware.md`](D_runtime_and_hardware.md) | D | Evaluation environment, hardware, and software versions |

## Related artefacts already in the repository

- **Dataset:** [`../data/measles_claims_dataset.csv`](../data/measles_claims_dataset.csv) — all 150 claims with the full seven-column schema
- **Raw results:** [`../evaluation/results/`](../evaluation/results/) — per-claim predictions, metrics, hallucination runs, confusion matrices
- **Code:** [`../pipeline/`](../pipeline/), [`../evaluation/`](../evaluation/)

> Items written as `EINFÜGEN: …` are placeholders to be filled from the local
> repository before final submission (exact version strings and per-run timings).
