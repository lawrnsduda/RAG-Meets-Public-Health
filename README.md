# RAG Meets Public Health

**Master Thesis Project:** Development and Evaluation of a Hybrid System for Automated Fact-Checking in Public Health

## Overview

**RAG Meets Public Health** is a prototype health claim verification system developed as part of a Master’s thesis. The project evaluates how reliably large language models can assess public health claims and whether a hybrid **Retrieval-Augmented Generation (RAG) + Natural Language Inference (NLI)** pipeline can improve factual accuracy, reduce hallucinations, and increase transparency.

The application allows users to enter a public health claim and receive a structured verdict based on evidence retrieved from trusted health and research sources.

## Main Idea

The project compares three pipeline settings:

- **Baseline** – LLM-only fact-checking without external retrieval
- **RAG-only** – evidence retrieval with FAISS + LLM-based verdict generation
- **RAG+NLI** – retrieval + NLI classification + LLM-based final verdict

The goal is to examine whether adding retrieval and claim–evidence reasoning leads to more robust and explainable fact-checking results.

## Features

- Streamlit-based user interface for interactive claim checking
- Configurable pipeline modes: Baseline, RAG-only, and RAG+NLI
- Evidence retrieval using **FAISS** and **sentence-transformers**
- Natural Language Inference with **DeBERTa v3 small**
- Local verdict generation via **Ollama** using **Qwen3 8B** or **Mistral 7B**
- Source filtering for trusted evidence providers
- Indexed evidence pipeline for public health knowledge
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
├── data/                # Data fetching scripts and FAISS index building
├── evaluation/          # Evaluation scripts / outputs
├── pipeline/            # Retrieval, NLI, and verdict generation logic
├── ui/                  # Streamlit UI components and styling
├── .env.example         # Example environment variables
├── Dockerfile           # Container setup
├── docker-compose.yml   # Docker orchestration
├── app.py               # Main Streamlit application
├── app_preview.py       # UI preview with dummy data
├── config.py            # Central project configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## How the Pipeline Works

1. **Claim input** – the user enters a health-related statement.
2. **Evidence retrieval** – relevant passages are retrieved from a FAISS index built from trusted sources.
3. **NLI classification** *(optional)* – each claim–evidence pair is classified as contradiction, entailment, or neutral.
4. **Verdict generation** – an LLM generates the final verdict:
   - `SUPPORTED`
   - `REFUTED`
   - `MISLEADING`
5. **Transparent output** – the interface shows the verdict, justification, and retrieved evidence passages.

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

## Example Use Case

**Input claim:**

> “The MMR vaccine has been scientifically proven to cause autism in children.”

**Pipeline output:**

- retrieves evidence from trusted medical and public health sources
- optionally classifies claim–evidence relations with NLI
- returns a structured verdict with a short justification

## Research Context

This repository supports a Master’s thesis investigating:

- the factual reliability of LLMs in public health fact-checking
- the effect of external evidence retrieval on model performance
- the contribution of NLI to explainability and claim verification
- the comparative performance of Baseline vs. RAG vs. RAG+NLI pipelines

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
