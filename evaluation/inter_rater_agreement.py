"""
Cohen's kappa between automatic and manual hallucination annotation.

Interpretation bands follow McHugh (2012), which provides a stricter
and more conservative classification than the older Landis & Koch (1977)
guidelines.

Reference:
    McHugh, M. L. (2012). Interrater reliability: the kappa statistic.
    Biochemia Medica, 22(3), 276-282. https://doi.org/10.11613/BM.2012.031

Usage:
    python -m evaluation.inter_rater_agreement path/to/sample.csv
"""

from __future__ import annotations
import csv
import sys
from pathlib import Path


VALID_CATEGORIES = {
    "hallucination_free",
    "minor_hallucination",
    "major_hallucination",
}


def load_sample(csv_path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            auto = row["auto_category"].strip()
            manual = row["manual_category"].strip()
            if not manual:
                continue
            if manual not in VALID_CATEGORIES:
                print(f"⚠  Skipping row {row.get('id', '?')}: "
                      f"invalid manual_category '{manual}'")
                continue
            pairs.append((auto, manual))
    return pairs


def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """kappa = (p_o - p_e) / (1 - p_e)."""
    if not pairs:
        return {"n": 0, "kappa": None}

    categories = sorted(set(VALID_CATEGORIES))
    n = len(pairs)

    agreements = sum(1 for a, m in pairs if a == m)
    p_o = agreements / n

    auto_counts = {c: 0 for c in categories}
    manual_counts = {c: 0 for c in categories}
    for a, m in pairs:
        auto_counts[a] = auto_counts.get(a, 0) + 1
        manual_counts[m] = manual_counts.get(m, 0) + 1

    p_e = sum(
        (auto_counts[c] / n) * (manual_counts[c] / n)
        for c in categories
    )

    if abs(1 - p_e) < 1e-9:
        kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)

    cm = {a: {m: 0 for m in categories} for a in categories}
    for a, m in pairs:
        cm[a][m] += 1

    return {
        "n": n,
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "kappa": round(kappa, 4),
        "confusion_matrix": cm,
    }


def interpret_kappa(kappa: float) -> str:
    """McHugh (2012) bands - stricter than Landis & Koch (1977)."""
    if kappa < 0:
        return "poor (worse than chance)"
    elif kappa < 0.21:
        return "none"
    elif kappa < 0.40:
        return "minimal"
    elif kappa < 0.60:
        return "weak"
    elif kappa < 0.80:
        return "moderate"
    elif kappa < 0.90:
        return "strong"
    else:
        return "almost perfect"


def main(csv_path_arg: str) -> None:
    path = Path(csv_path_arg)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    pairs = load_sample(path)
    if not pairs:
        print("No annotated rows found. Fill in 'manual_category' first.")
        return

    result = cohens_kappa(pairs)

    print(f"\nFile: {path.name}")
    print(f"Annotated pairs: {result['n']}")
    print(f"Observed agreement: {result['observed_agreement']:.2%}")
    print(f"Expected agreement: {result['expected_agreement']:.2%}")
    print(f"Cohen's kappa: {result['kappa']:.4f}")
    if result["kappa"] is not None:
        print(f"Interpretation: {interpret_kappa(result['kappa'])} agreement "
              f"(McHugh, 2012)")

    print("\nConfusion matrix (rows: auto, cols: manual):")
    cats = sorted(VALID_CATEGORIES)
    header = " " * 22 + "".join(f"{c[:12]:>14}" for c in cats)
    print(header)
    for a in cats:
        row_str = f"{a[:20]:>22}" + "".join(
            f"{result['confusion_matrix'][a][m]:>14d}" for m in cats
        )
        print(row_str)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation.inter_rater_agreement <sample.csv>")
        sys.exit(1)
    main(sys.argv[1])