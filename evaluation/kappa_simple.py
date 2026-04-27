"""
Cohen's Kappa zwischen auto_category und manual_category.
Toleriert verschiedene CSV-Encodings und Trennzeichen (, oder ;).
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

VALID_CATEGORIES = {"hallucination_free", "minor_hallucination", "major_hallucination"}

SHORT_TO_LONG = {
    "FREE": "hallucination_free",
    "MINOR": "minor_hallucination",
    "MAJOR": "major_hallucination",
    "free": "hallucination_free",
    "minor": "minor_hallucination",
    "major": "major_hallucination",
}


def normalize(val):
    if not val:
        return None
    v = str(val).strip()
    if v in VALID_CATEGORIES:
        return v
    return SHORT_TO_LONG.get(v)


def detect_delimiter(text):
    first_line = text.split("\n", 1)[0]
    if first_line.count(";") > first_line.count(","):
        return ";"
    return ","


def load_pairs(csv_path):
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "mac-roman"]
    for enc in encodings:
        try:
            with open(csv_path, "r", encoding=enc) as f:
                text = f.read()
            delim = detect_delimiter(text)
            print(f"CSV gelesen mit Encoding: {enc}, Trennzeichen: '{delim}'")

            pairs = []
            with open(csv_path, "r", encoding=enc) as f:
                reader = csv.DictReader(f, delimiter=delim)
                for row in reader:
                    auto = normalize(row.get("auto_category"))
                    manual = normalize(row.get("manual_category"))
                    if auto and manual:
                        pairs.append((auto, manual))
            return pairs
        except UnicodeDecodeError:
            continue
    return []


def cohens_kappa(pairs):
    if not pairs:
        return None
    categories = sorted(VALID_CATEGORIES)
    n = len(pairs)
    p_o = sum(1 for a, m in pairs if a == m) / n

    auto_counts = {c: 0 for c in categories}
    manual_counts = {c: 0 for c in categories}
    for a, m in pairs:
        auto_counts[a] += 1
        manual_counts[m] += 1

    p_e = sum((auto_counts[c] / n) * (manual_counts[c] / n) for c in categories)
    kappa = (p_o - p_e) / (1 - p_e) if abs(1 - p_e) > 1e-9 else 0.0

    cm = {a: {m: 0 for m in categories} for a in categories}
    for a, m in pairs:
        cm[a][m] += 1

    return {"n": n, "p_o": p_o, "p_e": p_e, "kappa": kappa, "cm": cm,
            "auto_counts": auto_counts, "manual_counts": manual_counts}


def interpret(k):
    if k < 0: return "poor (worse than chance)"
    if k < 0.21: return "none"
    if k < 0.40: return "minimal"
    if k < 0.60: return "weak"
    if k < 0.80: return "moderate"
    if k < 0.90: return "strong"
    return "almost perfect"


def main(path_arg):
    path = Path(path_arg)
    if not path.exists():
        print(f"Datei nicht gefunden: {path}")
        sys.exit(1)

    pairs = load_pairs(path)
    if not pairs:
        print("Keine annotierten Zeilen gefunden.")
        return

    r = cohens_kappa(pairs)
    print(f"\nDatei: {path.name}")
    print(f"Annotierte Paare: {r['n']}")
    print(f"Beobachtete Übereinstimmung: {r['p_o']:.2%}")
    print(f"Erwartete Übereinstimmung:   {r['p_e']:.2%}")
    print(f"Cohen's Kappa: {r['kappa']:.4f}")
    print(f"Interpretation: {interpret(r['kappa'])} agreement (McHugh, 2012)")

    print(f"\nVerteilung auto:   {dict(r['auto_counts'])}")
    print(f"Verteilung manual: {dict(r['manual_counts'])}")

    cats = sorted(VALID_CATEGORIES)
    print("\nConfusion Matrix (Zeilen: auto, Spalten: manual):")
    print(" " * 22 + "".join(f"{c[:14]:>16}" for c in cats))
    for a in cats:
        row = f"{a[:20]:>22}" + "".join(f"{r['cm'][a][m]:>16d}" for m in cats)
        print(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m evaluation.kappa_simple <csv-pfad>")
        sys.exit(1)
    main(sys.argv[1])
