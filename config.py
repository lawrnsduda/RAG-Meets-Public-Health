"""Configuration: paths, models, evaluation labels."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EVIDENECE_DIR = DATA_DIR / "evidence"
FAISS_INDEX_PATH = DATA_DIR / "evidence" / "index.faiss"
CLAIMS_CSV = DATA_DIR / "measles_claims_dataset.csv"

# Abstention is orthogonal to the mode and handled as a separate flag.
PIPELINE_MODES = {
    "Baseline": {
        "description": "LLM only - no retrieval, no NLI",
        "use_retrieval": False,
        "use_nli": False,
    },
    "RAG-only": {
        "description": "FAISS retrieval + LLM, no NLI",
        "use_retrieval": True,
        "use_nli": False,
    },
    "RAG+NLI": {
        "description": "Full pipeline - retrieval + LLM + NLI",
        "use_retrieval": True,
        "use_nli": True,
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K = 5

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
NLI_LABELS = ["contradiction", "entailment", "neutral"]

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_MISTRAL_MODEL = "mistral:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

EVIDENCE_SOURCES = {
    "WHO":    {"enabled": True, "label": "WHO Global Health Observatory"},
    "CDC":    {"enabled": True, "label": "CDC Data & Statistics"},
    "ECDC":   {"enabled": True, "label": "ECDC Surveillance Atlas"},
    "OWID":   {"enabled": True, "label": "Our World in Data"},
    "PubMed": {"enabled": True, "label": "NCBI E-utilities"},
}

EVAL_LABELS = ["SUPPORTED", "REFUTED", "MISLEADING"]

# Prediction-only label; ground truth remains 3-class.
ABSTAIN_LABEL = "NOT_ENOUGH_EVIDENCE"
EVAL_LABELS_WITH_ABSTAIN = EVAL_LABELS + [ABSTAIN_LABEL]