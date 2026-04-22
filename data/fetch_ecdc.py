"""
Fetch measles-related content from ECDC via web scraping.
"""

import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "evidence" / "ecdc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ECDC_PAGES = [
    {
        "url": "https://www.ecdc.europa.eu/en/measles",
        "label": "ECDC — Measles overview",
    },
    {
        "url": "https://www.ecdc.europa.eu/en/measles/facts",
        "label": "ECDC — Measles factsheet",
    },
]

HEADERS = {
    "User-Agent": "HealthClaimChecker/1.0 (academic research)"
}


def fetch_page(url: str, label: str) -> dict:
    print(f"  Fetching {label}…")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    main = soup.find("main") or soup.find("article") or soup
    paragraphs = [
        p.get_text(strip=True)
        for p in main.find_all("p")
        if p.get_text(strip=True)
    ]

    return {
        "source": "ECDC",
        "url": url,
        "label": label,
        "paragraphs": paragraphs,
        "full_text": "\n\n".join(paragraphs),
    }


def run():
    for page in ECDC_PAGES:
        try:
            data = fetch_page(page["url"], page["label"])
            slug = page["url"].rstrip("/").split("/")[-1]
            out_path = OUTPUT_DIR / f"{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(
                f"  → Saved {len(data['paragraphs'])} paragraphs"
                f" to {out_path}"
            )
        except Exception as e:
            print(f"  ✗ Failed: {page['label']} — {e}")


if __name__ == "__main__":
    run()
