"""
Configuration for Health FactChecker project.

Central place for settings, paths, model names, and API keys so the
project works the same on my laptop and on the FH server.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# PROJECT_ROOT is the folder where this config.py "lives". All other
# paths are relative to it so the project works regardless of where it's cloned.
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENECE_DIR = DATA_DIR / "evidence"
FAISS_INDEX_PATH = DATA_DIR / "evidence" / "index.faiss"
CLAIMS_CSV = DATA_DIR / "measles_claims_dataset.csv"

PIPELINE_MODES = {
    "Baseline": {
        "description": "LLM only - no retrieval, no NLI",
        "use_retrieval": False,
        "use_nli": False,
        "use_abstain": False,
    },
    "RAG-only": {
        "description": "FAISS retrieval + LLM, no NLI",
        "use_retrieval": True,
        "use_nli": False,
        "use_abstain": False,
    },
    "RAG+NLI": {
        "description": "Full pipeline - retrieval + LLM + NLI",
        "use_retrieval": True,
        "use_nli": True,
        "use_abstain": False,
    },
    # --- Abstain variants (4-class: adds NOT_ENOUGH_EVIDENCE) ---
    # Based on the AVeriTeC label scheme (Schlichtkrull et al., 2023).
    # The model can now say "not enough evidence" instead of guessing
    # a wrong REFUTED verdict when retrieval is weak.
    "RAG-only+Abstain": {
        "description": "RAG-only, but model can abstain (NOT_ENOUGH_EVIDENCE)",
        "use_retrieval": True,
        "use_nli": False,
        "use_abstain": True,
    },
    "RAG+NLI+Abstain": {
        "description": "RAG+NLI, but model can abstain (NOT_ENOUGH_EVIDENCE)",
        "use_retrieval": True,
        "use_nli": True,
        "use_abstain": True,
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384   # Fixed output dimension of MiniLM — do not change.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50          # Overlap so chunks don't cut sentences in half.
TOP_K = 5

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
# Label order must match the model's output neurons.
NLI_LABELS = ["contradiction", "entailment", "neutral"]

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_MISTRAL_MODEL = "mistral:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

LLM_TEMPERATURE = 0.1   # Low = deterministic verdicts.
LLM_MAX_TOKENS = 1024

# Loaded from .env so it doesn't get committed to git.
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

# Keys (WHO, CDC, …) are used as source tags in FAISS metadata and retriever filters.
EVIDENCE_SOURCES = {
    "WHO":    {"enabled": True, "label": "WHO Global Health Observatory"},
    "CDC":    {"enabled": True, "label": "CDC Data & Statistics"},
    "ECDC":   {"enabled": True, "label": "ECDC Surveillance Atlas"},
    "OWID":   {"enabled": True, "label": "Our World in Data"},
    "PubMed": {"enabled": True, "label": "NCBI E-utilities"},
}

# Ground-truth labels in the dataset (3-class).
EVAL_LABELS = ["SUPPORTED", "REFUTED", "MISLEADING"]

# Prediction labels in Abstain modes (4-class).
# NOT_ENOUGH_EVIDENCE is a *prediction-only* label — the dataset itself
# still contains only the 3 EVAL_LABELS, so no dataset rework is needed.
ABSTAIN_LABEL = "NOT_ENOUGH_EVIDENCE"
EVAL_LABELS_WITH_ABSTAIN = EVAL_LABELS + [ABSTAIN_LABEL]
