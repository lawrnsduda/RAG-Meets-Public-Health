"""
Evidence retrieval using FAISS (IndexFlatL2) and sentence-transformers.

This is the search part of the RAG pipeline – we take a health claim,
turn it into a vector, and then look up the most similar passages in our
pre-built FAISS index. The index contains chunks from WHO, CDC, PubMed etc.
"""

from __future__ import annotations 
import json
import numpy as np
from pathlib import Path
from config import ( 
    EMBEDDING_MODEL, 
    FAISS_INDEX_PATH, 
    TOP_K,
)

_model = None
_index = None
_metadata: list[dict] = []

def load_model():

    """Load the embedding model. Only does the actual loading on the first call,
    after that it just returns the cached instance."""

    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def load_index():

    """Load the FAISS index + the metadata JSON that goes with it.
    The .faiss file only has the vectors, so we need the .meta.json
    to know which text/source/url belongs to which vector position."""

    global _index, _metadata
    if _index is not None:
        return _index, _metadata
    
    import faiss
    index_path = Path(FAISS_INDEX_PATH)
    meta_path = index_path.with_suffix(".meta.json")

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run the indexing step first (see data/build_index.py)."
        )
    
    _index = faiss.read_index(str(index_path))
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
    else:
        _metadata = [{}] * _index.ntotal

    return _index, _metadata

def retrieve_evidence(
    claim: str,
    top_k: int = TOP_K,
    sources: list[str] | None = None,
) -> list[dict]:
    
    """
    Main function – takes a claim string and returns the top-k most
    relevant evidence passages from the index.
 
    If `sources` is set (e.g. ["WHO", "CDC"]) we only keep passages
    from those sources. In that case we fetch 3x more candidates from
    FAISS first, because a lot of them will get filtered out.
    """

    model = load_model()
    index, metadata = load_index()

    query_vec = model.encode([claim], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")

    search_k = top_k * 3 if sources else top_k
    distances, indices = index.search(query_vec, search_k)

    results: list[dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx] if idx < len(metadata) else {}
        source = meta.get("source", "unknown")

        if sources and source not in sources:
            continue

        results.append({
            "text": meta.get("text", ""),
            "source": source,
            "score": float(dist), 
            "url": meta.get("url", ""),
        })

        if len(results) >= top_k:
            break

    return results
   