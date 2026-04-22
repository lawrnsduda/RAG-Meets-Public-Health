"""
Fetch measles-related data from WHO Global Health Observatory
via OData API.
"""
import requests
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "evidence" / "who"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GHO_BASE = "https://ghoapi.azureedge.net/api"

INDICATORS = {
    "WHS4_100":        "Measles — number of reported cases",
    "VACCINES_COVERAGE_MCV1": "Measles-containing vaccine 1st dose coverage (%)",
    "VACCINES_COVERAGE_MCV2": "Measles-containing vaccine 2nd dose coverage (%)",
}


def fetch_indicator(code: str, label: str) -> list[dict]:
    url = f"{GHO_BASE}/{code}"
    params = {
        "$filter": "TimeDim ge 2010",
        "$orderby": "TimeDim desc",
        "$top": 500,
    }
    print(f"  Fetching {label} ({code})…")
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("value", [])
    except Exception as e:
        print(f"  ✗ Skipped {code}: {e}")
        return []


def run():
    for code, label in INDICATORS.items():
        records = fetch_indicator(code, label)
        if records:
            out_path = OUTPUT_DIR / f"{code}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"  → Saved {len(records)} records to {out_path}")
        else:
            print(f"  → No data for {code}, skipping.")


if __name__ == "__main__":
    run()