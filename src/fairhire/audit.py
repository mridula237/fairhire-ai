"""Command-line runner: audit a hiring file and print/save the results table.

Examples:
    PYTHONPATH=src python -m fairhire.audit data/demo_hiring_data.csv
    PYTHONPATH=src python -m fairhire.audit my_export.xlsx --attribute gender --reference pooled
    PYTHONPATH=src python -m fairhire.audit data/demo_hiring_data.csv --intersectional --out results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fairhire import schema
from fairhire.fairhire_stats import (
    SUMMARY_COLUMNS,
    add_intersection,
    audit_by_req,
    audit_funnel,
    audit_stage,
)


def load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}. See docs/data_dictionary.md")
    return df


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="FairHire AI funnel audit")
    ap.add_argument("file", type=Path)
    ap.add_argument("--attribute", default="race_ethnicity", help="column to audit (default race_ethnicity)")
    ap.add_argument("--intersectional", action="store_true", help="audit gender x race_ethnicity")
    ap.add_argument("--reference", default="highest", help="'highest' (default), 'pooled', or a group name")
    ap.add_argument("--min-n", type=int, default=schema.MIN_GROUP_N)
    ap.add_argument("--req", help="audit only this req_id")
    ap.add_argument("--by-req", action="store_true", help="also audit each requisition separately")
    ap.add_argument("--out", type=Path, help="save the full table as CSV")
    args = ap.parse_args(argv)

    df = load(args.file)
    if args.req:
        df = df[df["req_id"] == args.req]
    attr = args.attribute
    if args.intersectional:
        df = add_intersection(df)
        attr = "gender_x_race_ethnicity"

    full = audit_funnel(df, attr, args.reference, args.min_n)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(f"FairHire AI audit · {args.file.name} · attribute={attr} · reference={args.reference}")
    print("Verdicts use Holm-adjusted p-values across every group x stage test.\n")
    for stage in schema.STAGES[1:]:
        c = audit_stage(df, attr, stage, args.reference, args.min_n).attrs["chi_square"]
        t = full[full["stage"] == stage].set_index(attr)
        prev = schema.STAGES[schema.STAGES.index(stage) - 1]
        print(f"--- {prev} -> {stage}  (chi-square p = {c['p_value']:.2g}, dof = {c['dof']})")
        print(t[SUMMARY_COLUMNS[1:]].round(3).to_string(), "\n")

    flagged = full[full["verdict"] == "adverse impact indicated"]
    print("SUMMARY")
    if flagged.empty:
        print("No stage shows adverse impact that is both below four-fifths and statistically significant.")
    for _, r in flagged.iterrows():
        print(f"  {r['stage']}: {r[attr]}  AIR {r['air']:.2f} "
              f"(95% CI {r['air_ci_low']:.2f}-{r['air_ci_high']:.2f}), adjusted p = {r['p_adjusted']:.2g}")
    if args.by_req:
        per = audit_by_req(df, attr, args.reference, args.min_n)
        print("\nBY REQUISITION (each req audited separately; small reqs mostly get 'no verdict')")
        if per.empty:
            print("  No requisition has enough applicants for a verdict.")
        else:
            counts = per.groupby("req_id")["verdict"].apply(lambda v: (v == "adverse impact indicated").sum())
            for req, k in counts.items():
                title = df.loc[df["req_id"] == req, "job_title"].iloc[0]
                print(f"  {req} ({title}): {k} flag(s)")
            if args.out:
                per_path = args.out.with_name(args.out.stem + "_by_req.csv")
                per.to_csv(per_path, index=False)
                print(f"  Saved per-requisition table -> {per_path}")
    print("\nStatistical analysis only, not legal advice.")

    if args.out:
        full.to_csv(args.out, index=False)
        print(f"Saved full table -> {args.out}")


if __name__ == "__main__":
    main()
