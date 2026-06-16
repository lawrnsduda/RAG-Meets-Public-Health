# RAG Meets Public Health

**Master Thesis Project:** Development and Evaluation of a Hybrid System for
Automated Fact-Checking in Public Health

## Overview

**RAG Meets Public Health** is a prototype health-claim verification system
developed as part of a Master's thesis. The project evaluates how reliably
local large language models can assess public-health claims and whether a
hybrid **Retrieval-Augmented Generation (RAG) + Natural Language Inference
(NLI)** pipeline improves factual accuracy, reduces hallucinations, and
increases transparency over an LLM-only baseline.

The application lets users enter a public-health claim and receive a structured
verdict based on evidence retrieved from trusted health and research sources.

> **Repository scope.** This repository is the reproducibility appendix of the
> thesis. Alongside the source code it ships the evaluation dataset, prompt
> templates, configuration, and the generated results
> (`evaluation/results/thesis_output/` and the raw run JSONs).

## Main Idea

The project compares three pipeline settings, each evaluated with and without an
abstain option:

- **Baseline** – LLM-only fact-checking without external retrieval
- **RAG-only** – evidence retrieval with FAISS + LLM-based verdict generation
- **RAG+NLI** – retrieval + NLI classification + LLM-based final verdict

Each mode can additionally allow the model to abstain via a
`NOT_ENOUGH_EVIDENCE` verdict, yielding **six experimental conditions** in total
(3 modes × 2 abstain settings), each run with two local LLMs (Qwen3 8B and
Mistral 7B).

## Verdict Labels

The system emits three core labels, plus one prediction-only abstain label:

| Output label (code & UI) | Thesis figures | Meaning |
|---|---|---|
| `SUPPORTED` | True | evidence confirms the claim |
| `REFUTED` | False | evidence contradicts the claim |
| `MISLEADING` | Misleading | partially true but cherry-picked / missing context |
| `NOT_ENOUGH_EVIDENCE` | Unverifiable | abstain — only in abstain conditions |

`SUPPORTED` / `REFUTED` / `MISLEADING` are the canonical labels used throughout
the code; some thesis figures use the plain-language equivalents above. Ground
truth remains three-class; `NOT_ENOUGH_EVIDENCE` is a prediction-only label.

## Evidence Sources

The evaluated FAISS index comprises **723 passages from four sources**:

| Source | Passages |
|---|---|
| WHO – Global Health Observatory | 500 |
| PubMed / NCBI | 169 |
| CDC – Data & Statistics | 38 |
| ECDC – Surveillance | 16 |

> **Transparency note.** Our World in Data (OWID) and IHME/GBD were specified in
> the system design but are **not** part of the realised index: the OWID fetch
> returned no usable passages, and the IHME/GBD ingestion step was not
> implemented. The corresponding fetch script is retained for documentation.

## Key Findings

Evaluated on 150 annotated measles claims (3-class ground truth):

| Condition | Qwen3 F1 | Mistral F1 |
|---|---|---|
| **Baseline** | 0.517 | **0.626** |
| Baseline+Abstain | 0.529 | 0.507 |
| RAG-only | 0.414 | 0.536 |
| RAG-only+Abstain | 0.350 | 0.469 |
| RAG+NLI | 0.442 | 0.528 |
| RAG+NLI+Abstain | 0.315 | 0.449 |

- **Contrary to the initial hypothesis, retrieval (RAG) did not improve
  performance over the LLM-only Baseline.** The best result was the **Mistral 7B
  Baseline (macro-F1 = 0.626)**; adding retrieval and NLI lowered macro-F1 for
  both models.
- The **NLI cross-encoder produced ~87% neutral labels** and was largely
  uninformative in this configuration. Its three-class taxonomy
  (entailment / contradiction / neutral) has **no explicit "not enough
  information" (NEI) class**, which is a structural driver of misclassification.
- The explicit abstain option traded coverage for selective accuracy but did not
  raise overall macro-F1.
- The system is therefore positioned as **human-in-the-loop triage support, not
  an autonomous fact-checker.**

Full per-class, selective-classification, and hallucination tables (plus
confusion-matrix figures) are in `evaluation/results/thesis_output/`.

## Features

- Streamlit-based UI for interactive claim checking
- Configurable pipeline modes: Baseline, RAG-only, RAG+NLI
- Orthogonal **abstain mode** with a dedicated `NOT_ENOUGH_EVIDENCE` label
- Evidence retrieval with **FAISS** + **sentence-transformers** (all-MiniLM-L6-v2)
- NLI with the **DeBERTa-v3-small** cross-encoder
- Local verdict generation via **Ollama** (**Qwen3 8B** or **Mistral 7B**)
- Automated **hallucination analysis** via per-sentence NLI grounding checks
- Cohen's kappa **inter-rater agreement** tooling for manual validation
- Selective-prediction metrics: coverage, selective accuracy, per-class abstention

## Project Structure

```text
RAG-Meets-Public-Health/
├── data/                              # Fetch scripts + FAISS index building
│   └── measles_claims_dataset.csv     # 150 annotated evaluation claims
├── evaluation/                        # Evaluation, hallucination & agreement tooling
│   ├── evaluate.py                    # Runs all 6 conditions and reports metrics
│   ├── hallucination_analysis.py      # Post-hoc per-sentence hallucination check
│   ├── inter_rater_agreement.py       # Cohen's kappa (auto vs. manual)
│   ├── build_thesis_tables.py         # Generates thesis tables + figures
│   └── results/                       # Committed thesis results (appendix)
│       ├── eval_*.json                # Raw per-claim predictions + metrics
│       ├── hallucination_*.json       # Per-sentence hallucination runs
│       ├── hallucination_samples_*/   # Stratified manual-annotation samples
│       └── thesis_output/             # Tables (md) + confusion-matrix figures
├── pipeline/                          # retrieval / nli / verdict / hallucination
├── ui/                                # Streamlit UI components and styling
├── Hallucinations_manual.csv          # Manual hallucination annotations
├── app.py / app_preview.py / config.py
├── Dockerfile / docker-compose.yml / requirements.txt
└── README.md
```

## How the Pipeline Works

1. **Claim input** – the user enters a health-related statement.
2. **Evidence retrieval** – relevant passages are retrieved from the FAISS index.
3. **NLI classification** *(optional)* – each claim–evidence pair is labelled
   contradiction / entailment / neutral.
4. **Verdict generation** – the LLM returns `SUPPORTED`, `REFUTED`,
   `MISLEADING`, or (in abstain mode) `NOT_ENOUGH_EVIDENCE`.
5. **Transparent output** – verdict, justification, and the retrieved evidence
   passages are shown.

## Installation

```bash
git clone https://github.com/lawrnsduda/RAG-Meets-Public-Health.git
cd RAG-Meets-Public-Health
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # set NCBI_API_KEY and OLLAMA_BASE_URL
```

## Prepare the Evidence Index

```bash
python -m data.fetch_all
python data/build_index.py
```

> The OWID fetch yields no usable passages; the index build completes from the
> four working sources (WHO, PubMed, CDC, ECDC) regardless.

## Run the App

```bash
streamlit run app.py          # full pipeline
streamlit run app_preview.py  # UI preview with dummy data
```

## Docker

```bash
docker compose up --build
```

## Evaluation

```bash
python -m evaluation.evaluate                 # all 6 conditions per LLM -> results/*.json
python -m evaluation.hallucination_analysis   # per-sentence hallucination check
python -m evaluation.inter_rater_agreement evaluation/results/.../sample_<condition>.csv
python -m evaluation.build_thesis_tables      # regenerate thesis_output/ tables + figures
```

## Research Context

This repository supports a Master's thesis investigating the factual reliability
of local LLMs in public-health fact-checking, the (null) effect of retrieval,
the limited contribution of a three-class NLI stage without an NEI label, and
the impact of an explicit abstain option on selective accuracy and hallucination
rates.

It is **not** a certified medical decision-support system and must not replace
professional medical advice.

## AI Disclosure

Parts of the code and documentation were developed with AI assistance
(Claude Opus). See §5.4.3 of the thesis for the full disclosure.

## Author

**Laurence Douda** — Master's thesis, FH St. Pölten
