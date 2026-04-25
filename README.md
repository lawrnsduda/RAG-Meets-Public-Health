# RAG Meets Public Health

**Master Thesis Project:** Development and Evaluation of a Hybrid System for Automated Fact-Checking in Public Health

## Overview

**RAG Meets Public Health** is a prototype health claim verification system developed as part of a Master’s thesis. The project evaluates how reliably large language models can assess public health claims and whether a hybrid **Retrieval-Augmented Generation (RAG) + Natural Language Inference (NLI)** pipeline can improve factual accuracy, reduce hallucinations, and increase transparency.

The application allows users to enter a public health claim and receive a structured verdict based on evidence retrieved from trusted health and research sources.

## Main Idea

The project compares three pipeline settings, each evaluated with and without an abstain option:

- **Baseline** – LLM-only fact-checking without external retrieval
- **RAG-only** – evidence retrieval with FAISS + LLM-based verdict generation
- **RAG+NLI** – retrieval + NLI classification + LLM-based final verdict

Each mode can additionally allow the model to abstain via a `NOT_ENOUGH_EVIDENCE` verdict, yielding **six experimental conditions** in total (3 modes × 2 abstain settings). The goal is to examine whether adding retrieval, claim–evidence reasoning, and a calibrated abstain option leads to more robust and explainable fact-checking results.

## Features

- Streamlit-based user interface for interactive claim checking
- Configurable pipeline modes: Baseline, RAG-only, and RAG+NLI
- Orthogonal **abstain mode** with a dedicated `NOT_ENOUGH_EVIDENCE` label
- Evidence retrieval using **FAISS** and **sentence-transformers**
- Natural Language Inference with **DeBERTa v3 small**
- Local verdict generation via **Ollama** using **Qwen3 8B** or **Mistral 7B**
- Automated **hallucination analysis** via per-sentence NLI grounding checks
- Cohen's kappa **inter-rater agreement** tooling for manual validation
- Selective-prediction metrics: coverage, selective accuracy, and per-class abstention rate
- Source filtering for trusted evidence providers
- Modular structure for data collection, retrieval, inference, evaluation, and UI

## Evidence Sources

The system is designed to work with trusted public health and research sources, including:

- **WHO** – Global Health Observatory
- **CDC** – Data and Statistics / public health content
- **ECDC** – surveillance and public health information
- **Our World in Data (OWID)**
- **PubMed / NCBI**

## Project Structure

```text
RAG-Meets-Public-Health/
├── data/                              # Data fetching scripts and FAISS index building
│   └── measles_claims_dataset.csv     # Annotated evaluation claims
├── evaluation/                        # Evaluation, hallucination & agreement tooling
│   ├── evaluate.py                    # Runs all 6 conditions and reports metrics
│   ├── hallucination_analysis.py      # Post-hoc per-sentence hallucination check
│   ├── inter_rater_agreement.py       # Cohen's kappa (auto vs. manual)
│   └── results/                       # Generated JSON / CSV outputs
├── pipeline/                          # Retrieval, NLI, verdict, hallucination logic
│   ├── retrieval.py
│   ├── nli.py
│   ├── verdict.py
│   └── hallucination.py
├── ui/                                # Streamlit UI components and styling
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── app.py                             # Main Streamlit application
├── app_preview.py                     # UI preview with dummy data
├── config.py                          # Central project configuration
├── requirements.txt
└── README.md
```

## How the Pipeline Works

1. **Claim input** – the user enters a health-related statement.
2. **Evidence retrieval** – relevant passages are retrieved from a FAISS index built from trusted sources.
3. **NLI classification** *(optional)* – each claim–evidence pair is classified as contradiction, entailment, or neutral.
4. **Verdict generation** – an LLM generates the final verdict:
   - `SUPPORTED`
   - `REFUTED`
   - `MISLEADING`
   - `NOT_ENOUGH_EVIDENCE` *(only when abstain mode is enabled)*
5. **Transparent output** – the interface shows the verdict, justification, and retrieved evidence passages.

The abstain option is orthogonal to the pipeline mode and is enabled in the UI by default. Ground-truth labels for evaluation remain three-class (`SUPPORTED` / `REFUTED` / `MISLEADING`); `NOT_ENOUGH_EVIDENCE` is a prediction-only label.

## Tech Stack

- **Python**
- **Streamlit**
- **FAISS**
- **sentence-transformers**
- **transformers**
- **PyTorch**
- **Ollama**
- **Docker / Docker Compose**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lawrnsduda/RAG-Meets-Public-Health.git
cd RAG-Meets-Public-Health
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file based on `.env.example`:

```env
NCBI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
```

## Prepare the Evidence Index

Before running the full application, fetch the evidence and build the vector index:

```bash
python -m data.fetch_all
python data/build_index.py
```

This will:

- collect evidence from the configured public health sources
- convert the data into text passages
- embed the passages
- build a FAISS index
- store metadata for retrieval

## Run the App

### Main application

```bash
streamlit run app.py
```

### UI preview only

```bash
streamlit run app_preview.py
```

The preview version is useful if you want to test the interface without running the full retrieval and model pipeline.

## Docker

If you prefer running the project in containers:

```bash
docker compose up --build
```

## Evaluation

The evaluation script runs every claim through all six conditions (3 modes × abstain on/off) for each LLM and writes a timestamped JSON to `evaluation/results/`:

```bash
python -m evaluation.evaluate
```

Reported metrics include accuracy, macro-F1, and per-class precision / recall / F1. For abstain conditions the script additionally computes **selective metrics**: coverage, selective accuracy, and per-class abstention rate.

### Hallucination Analysis

A post-hoc per-sentence NLI check classifies each generated justification as `hallucination_free`, `minor_hallucination`, or `major_hallucination` and produces stratified samples for manual annotation:

```bash
python -m evaluation.hallucination_analysis
# or target a specific run:
python -m evaluation.hallucination_analysis evaluation/results/eval_qwen3_<timestamp>.json
```

### Inter-rater Agreement

Cohen's kappa between automatic and manual hallucination annotations, with interpretation bands following McHugh (2012):

```bash
python -m evaluation.inter_rater_agreement evaluation/results/.../sample_<condition>.csv
```

## Example Use Case

**Input claim:**

> “The MMR vaccine has been scientifically proven to cause autism in children.”

**Pipeline output:**

- retrieves evidence from trusted medical and public health sources
- optionally classifies claim–evidence relations with NLI
- returns a structured verdict with a short justification, optionally abstaining via `NOT_ENOUGH_EVIDENCE`

## Research Context

This repository supports a Master’s thesis investigating:

- the factual reliability of LLMs in public health fact-checking
- the effect of external evidence retrieval on model performance
- the contribution of NLI to explainability and claim verification
- the comparative performance of Baseline vs. RAG vs. RAG+NLI pipelines
- the impact of an explicit abstain option on selective accuracy and hallucination rates

## Current Status

This repository is a prototype and research project. It is intended for:

- experimentation
- evaluation in an academic setting
- exploring explainable AI-supported fact-checking workflows

It is **not** a certified medical decision-support system and should not be used as a substitute for professional medical advice.

## Possible Future Improvements

- broader source coverage
- stronger retrieval ranking
- more advanced dashboards for evaluation results

## Author

**Laurence Douda**  
Master’s thesis project, FH St. Pölten
& Claude Opus 4.6 & 4.7
