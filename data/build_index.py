"""
Build the FAISS index from all fetched evidence files.

Run this after data/fetch_all.py. It reads every JSON file from
data/evidence/, converts the records to text chunks, embeds them
with sentence-transformers, and saves a FAISS index + a metadata
JSON file (same path, .meta.json extension).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DATA_DIR,
)

EVIDENCE_DIR = DATA_DIR / "evidence"


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _passage(text: str, source: str, url: str) -> list[dict]:
    return [{"text": c, "source": source, "url": url} for c in _chunk(text) if len(c.split()) >= 10]


def load_who() -> list[dict]:
    out = []
    who_dir = EVIDENCE_DIR / "who"
    if not who_dir.exists():
        return out
    labels = {
        "WHS4_100": "measles reported cases",
        "MCV1": "measles vaccine 1st-dose coverage (%)",
        "MCV2": "measles vaccine 2nd-dose coverage (%)",
    }
    for path in who_dir.glob("*.json"):
        code = path.stem
        label = labels.get(code, code)
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
            for r in records:
                country = r.get("SpatialDim", "")
                year = r.get("TimeDim", "")
                value = r.get("NumericValue")
                if not country or not year or value is None:
                    continue
                text = f"In {year}, {country} had a {label} of {value}."
                out.append({"text": text, "source": "WHO", "url": "https://www.who.int/data/gho"})
        except Exception as e:
            print(f"  Warning: {path.name}: {e}")
    return out


def load_text_pages(source_dir: Path, source_name: str) -> list[dict]:
    out = []
    if not source_dir.exists():
        return out
    for path in source_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            url = data.get("url", "")
            paragraphs = data.get("paragraphs", [])
            if paragraphs:
                for para in paragraphs:
                    out.extend(_passage(para, source_name, url))
            else:
                out.extend(_passage(data.get("full_text", ""), source_name, url))
        except Exception as e:
            print(f"  Warning: {path.name}: {e}")
    return out


def load_owid() -> list[dict]:
    out = []
    owid_dir = EVIDENCE_DIR / "owid"
    if not owid_dir.exists():
        return out
    articles_path = owid_dir / "search_articles.json"
    if articles_path.exists():
        try:
            articles = json.loads(articles_path.read_text(encoding="utf-8"))
            for a in articles:
                text = f"{a.get('title', '')}. {a.get('excerpt', '')}".strip()
                url = a.get("url", "https://ourworldindata.org")
                out.extend(_passage(text, "OWID", url))
        except Exception as e:
            print(f"  Warning: OWID articles: {e}")
    return out


def load_pubmed() -> list[dict]:
    out = []
    path = EVIDENCE_DIR / "pubmed" / "abstracts.json"
    if not path.exists():
        return out
    try:
        articles = json.loads(path.read_text(encoding="utf-8"))
        for a in articles:
            text = f"{a.get('title', '')}. {a.get('abstract', '')}".strip()
            url = a.get("url", "https://pubmed.ncbi.nlm.nih.gov/")
            out.extend(_passage(text, "PubMed", url))
    except Exception as e:
        print(f"  Warning: PubMed: {e}")
    return out


def build_index() -> None:
    print("=" * 60)
    print("Health Factchecker — Building FAISS index")
    print("=" * 60)

    all_passages: list[dict] = []

    steps = [
        ("[1/5] WHO",    load_who),
        ("[2/5] CDC",    lambda: load_text_pages(EVIDENCE_DIR / "cdc", "CDC")),
        ("[3/5] ECDC",   lambda: load_text_pages(EVIDENCE_DIR / "ecdc", "ECDC")),
        ("[4/5] OWID",   load_owid),
        ("[5/5] PubMed", load_pubmed),
    ]

    for label, fn in steps:
        print(f"\n{label}")
        passages = fn()
        print(f"  → {len(passages)} passages")
        all_passages.extend(passages)

    print(f"\nTotal passages: {len(all_passages)}")

    if not all_passages:
        print("\nNo passages found. Run `python -m data.fetch_all` first.")
        sys.exit(1)

    print("\nEmbedding passages (this may take a few minutes)…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [p["text"] for p in all_passages]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")

    print("\nBuilding FAISS index…")
    import faiss
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    index_path = Path(FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    meta_path = index_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(all_passages, ensure_ascii=False), encoding="utf-8")

    print(f"\nIndex  → {index_path}  ({index.ntotal} vectors)")
    print(f"Meta   → {meta_path}")
    print("Done.")


if __name__ == "__main__":
    build_index()
