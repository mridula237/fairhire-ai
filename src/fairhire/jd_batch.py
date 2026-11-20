"""Score a folder of job descriptions (.txt or .docx) in one go.

    PYTHONPATH=src python -m fairhire.jd_batch path/to/postings --out jd_results.csv

Prints one row per file plus totals (how many lack a pay range, most common flags), which is
what the Week 4 "10 public Chicago postings" benchmark needs. Don't put employer names in
public posts; the filenames are yours to choose (e.g. jd01.txt).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from fairhire.jd_audit import audit_jd, read_docx


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx(path.read_bytes())
    return path.read_text(encoding="utf-8", errors="replace")


def score_folder(folder: Path) -> tuple[pd.DataFrame, Counter]:
    rows, flag_counter = [], Counter()
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in {".txt", ".docx"}:
            continue
        r = audit_jd(load_text(p))
        flag_counter.update(h.kind for h in r.flags)
        rows.append({
            "file": p.name,
            "score": r.score,
            "gender_coding": r.gender_coding,
            "masculine_words": len(r.masculine),
            "feminine_words": len(r.feminine),
            "flags": len(r.flags),
            "high_severity_flags": sum(h.severity == "high" for h in r.flags),
            "flag_ids": ", ".join(sorted({h.kind for h in r.flags})),
            "fk_grade": r.readability["fk_grade"],
            "words": r.readability["words"],
            "has_pay_range": r.has_pay_range,
        })
    return pd.DataFrame(rows), flag_counter


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FairHire AI: batch job-description audit")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    df, flags = score_folder(args.folder)
    if df.empty:
        raise SystemExit("No .txt or .docx files found.")
    pd.set_option("display.width", 200)
    print(df.drop(columns="flag_ids").to_string(index=False))
    n = len(df)
    print(f"\n{n} postings · median score {df['score'].median():.0f}/100 · "
          f"{(~df['has_pay_range']).sum()} of {n} have no pay range · "
          f"{(df['flags'] > 0).sum()} of {n} have at least one flagged phrase · "
          f"median reading grade {df['fk_grade'].median():.1f}")
    coding = df["gender_coding"].value_counts().to_dict()
    print("Gender coding: " + ", ".join(f"{k}: {v}" for k, v in coding.items()))
    if flags:
        print("Most common flags: " + ", ".join(f"{k} ({v})" for k, v in flags.most_common(5)))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
