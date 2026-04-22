"""
Fetch measles-related abstracts from PubMed via NCBI E-utilities.
"""

import requests
import json
import time
from pathlib import Path
from config import NCBI_API_KEY

OUTPUT_DIR = Path(__file__).parent / "evidence" / "pubmed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

QUERIES = [
    "measles vaccine safety efficacy",
    "measles transmission epidemiology",
    "measles MMR autism",
    "measles outbreak Europe",
    "measles complications mortality",
]

MAX_RESULTS_PER_QUERY = 20


def search_pubmed(query: str) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": MAX_RESULTS_PER_QUERY,
        "retmode": "json",
        "sort": "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(EFETCH_URL, params=params, timeout=30)
    resp.raise_for_status()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "xml")

    articles = []
    for article in soup.find_all("PubmedArticle"):
        pmid = article.find("PMID")
        title = article.find("ArticleTitle")
        abstract = article.find("AbstractText")

        articles.append({
            "source": "PubMed",
            "pmid": pmid.text if pmid else "",
            "title": title.text if title else "",
            "abstract": abstract.text if abstract else "",
            "url": (
                f"https://pubmed.ncbi.nlm.nih.gov/{pmid.text}/"
                if pmid
                else ""
            ),
        })

    return articles


def run():
    all_articles = []
    seen_pmids = set()

    for query in QUERIES:
        print(f"  Searching: {query}")
        pmids = search_pubmed(query)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)

        if new_pmids:
            articles = fetch_abstracts(new_pmids)
            all_articles.extend(articles)
            print(f"  → Fetched {len(articles)} abstracts")

        time.sleep(0.4)

    out_path = OUTPUT_DIR / "abstracts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    print(
        f"\n  Total: {len(all_articles)} unique abstracts"
        f" → {out_path}"
    )


if __name__ == "__main__":
    run()