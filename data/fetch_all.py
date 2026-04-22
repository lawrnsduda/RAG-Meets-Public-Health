"""Run all evidence fetchers sequentially."""

from data import (
    fetch_who,
    fetch_cdc,
    fetch_ecdc,
    fetch_owid,
    fetch_pubmed,
)


def main():
    print("=" * 60)
    print("Health Claim Checker — Evidence Fetcher")
    print("=" * 60)

    print("\n[1/5] WHO Global Health Observatory")
    fetch_who.run()

    print("\n[2/5] CDC")
    fetch_cdc.run()

    print("\n[3/5] ECDC")
    fetch_ecdc.run()

    print("\n[4/5] Our World in Data")
    fetch_owid.run()

    print("\n[5/5] PubMed")
    fetch_pubmed.run()

    print("\n" + "=" * 60)
    print("All evidence fetched. Next step: build FAISS index.")
    print("=" * 60)


if __name__ == "__main__":
    main()