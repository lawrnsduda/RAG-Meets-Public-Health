"""
Fetch measles-related data from Our World in Data (OWID).
"""

import requests
import pandas as pd
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "evidence" / "owid"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OWID_MEASLES_CSV = (
    "https://raw.githubusercontent.com/owid/etl/master/"
    "etl/steps/data/garden/who/2024-12-02/measles.csv"
)

OWID_SEARCH_URL = "https://search.owid.io/search"


def fetch_csv_data() -> pd.DataFrame:
    print("  Fetching OWID measles CSV…")
    try:
        df = pd.read_csv(OWID_MEASLES_CSV)
        out_path = OUTPUT_DIR / "measles_data.csv"
        df.to_csv(out_path, index=False)
        print(f"  → Saved {len(df)} rows to {out_path}")
        return df
    except Exception as e:
        print(f"  ✗ CSV fetch failed: {e}")
        return pd.DataFrame()


def fetch_search_articles(
    query: str = "measles vaccination",
) -> list[dict]:
    print(f"  Searching OWID for '{query}'…")
    try:
        resp = requests.get(
            OWID_SEARCH_URL,
            params={"q": query},
            timeout=15,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])

        articles = []
        for hit in hits[:10]:
            articles.append({
                "source": "OWID",
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "excerpt": hit.get("excerpt", ""),
            })

        out_path = OUTPUT_DIR / "search_articles.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"  → Saved {len(articles)} articles to {out_path}")
        return articles
    except Exception as e:
        print(f"  ✗ Search failed: {e}")
        return []


def run():
    fetch_csv_data()
    fetch_search_articles("measles vaccination coverage")
    fetch_search_articles("measles outbreak")


if __name__ == "__main__":
    run()