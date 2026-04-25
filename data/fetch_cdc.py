"""
Fetch measles-related data from CDC via the Socrata Open Data API (SODA).
No API key required. Documentation: https://dev.socrata.com/
"""
import requests
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "evidence" / "cdc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CDC_DATASETS = [
    {
        "label": "NNDSS — Measles weekly cases",
        "url": "https://data.cdc.gov/resource/x9gk-5huc.json",
        "params": {
            "$where": "disease = 'Measles'",
            "$order": "mmwr_year DESC",
            "$limit": 500,
        },
    },
    {
        "label": "Vaccination coverage — MMR (children 19-35 months)",
        "url": "https://data.cdc.gov/resource/fhky-rtsk.json",
        "params": {
            "$where": "vaccine = 'MMR'",
            "$order": "survey_years DESC",
            "$limit": 200,
        },
    },
]

CDC_PAGES = [
    {
        "url": "https://www.cdc.gov/measles/about/index.html",
        "label": "Measles — About",
    },
    {
        "url": "https://www.cdc.gov/measles/signs-symptoms/index.html",
        "label": "Measles — Signs & Symptoms",
    },
    {
        "url": "https://www.cdc.gov/measles/vaccines/index.html",
        "label": "Measles — Vaccination",
    },
    {
        "url": "https://www.cdc.gov/measles/causes/",
        "label": "Measles — Transmission",
    },
]

HEADERS = {
    "User-Agent": "HealthClaimChecker/1.0 (academic research)"
}


def fetch_soda_dataset(url: str, label: str, params: dict) -> list[dict]:
    print(f"  Fetching {label}…")
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        print(f"  → Got {len(data)} records")
        return data
    except Exception as e:
        print(f"  ✗ Failed: {label} — {e}")
        return []


def fetch_page_text(url: str, label: str) -> dict:
    from bs4 import BeautifulSoup

    print(f"  Fetching {label}…")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        main = (
            soup.find("main")
            or soup.find("div", {"role": "main"})
            or soup
        )
        paragraphs = [
            p.get_text(strip=True)
            for p in main.find_all("p")
            if p.get_text(strip=True)
        ]

        return {
            "source": "CDC",
            "url": url,
            "label": label,
            "paragraphs": paragraphs,
            "full_text": "\n\n".join(paragraphs),
        }
    except Exception as e:
        print(f"  ✗ Failed: {label} — {e}")
        return {}


def run():
    print("  — SODA API (structured data) —")
    for ds in CDC_DATASETS:
        data = fetch_soda_dataset(ds["url"], ds["label"], ds["params"])
        if data:
            slug = ds["url"].split("/")[-1].replace(".json", "")
            out_path = OUTPUT_DIR / f"soda_{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"  → Saved to {out_path}")

    print("  — Fact sheets (text content) —")
    for page in CDC_PAGES:
        data = fetch_page_text(page["url"], page["label"])
        if data:
            slug = page["url"].rstrip("/").split("/")[-1]
            out_path = OUTPUT_DIR / f"page_{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(
                f"  → Saved {len(data.get('paragraphs', []))} "
                f"paragraphs to {out_path}"
            )


if __name__ == "__main__":
    run()