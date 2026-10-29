"""FairHire statistics engine (Week 2, part 1): selection rates and adverse impact ratios.

Definitions
-----------
* Selection rate into stage k = applicants who reached stage k / applicants who reached stage k-1.
* Adverse impact ratio (AIR) = group's rate / reference rate.
* Reference options:
    - "highest": highest rate among eligible groups (29 CFR 1607.4(D), NYC LL144). Eligible =
      known response and n >= min_n, so a tiny lucky group can't become the benchmark.
    - "pooled":  each group vs. all OTHER known-response applicants pooled. More stable.
    - any group name, e.g. "White": compare everyone to that group.

Coming in the rest of Week 2: z-test, Fisher's exact, chi-square, SD test, Wilson CIs.
"""

from __future__ import annotations

import pandas as pd

from fairhire import schema

FOUR_FIFTHS = 0.80
LL144_MIN_SHARE = 0.02  # NYC LL144: categories under 2% of data may be excluded from impact ratios

_ORDER = {s: i for i, s in enumerate(schema.STAGES)}


def add_intersection(df: pd.DataFrame, a: str = "gender", b: str = "race_ethnicity") -> pd.DataFrame:
    """Add an intersectional column like 'Female | Hispanic or Latino' (LL144 style).
    If either part is a non-response value, the combination is 'Unknown'."""
    out = df.copy()
    unknown = out[a].isin(schema.NON_RESPONSE_VALUES) | out[b].isin(schema.NON_RESPONSE_VALUES)
    out[f"{a}_x_{b}"] = (out[a] + " | " + out[b]).where(~unknown, "Unknown")
    return out


def stage_rates(df: pd.DataFrame, attribute: str, stage: str) -> pd.DataFrame:
    """Selection rate into `stage` by group: n (at risk), selected, rate."""
    if stage not in _ORDER or _ORDER[stage] == 0:
        raise ValueError(f"stage must be one of {schema.STAGES[1:]}")
    k = _ORDER[stage]
    reached = df["stage_reached"].map(_ORDER)
    if reached.isna().any():
        bad = sorted(df.loc[reached.isna(), "stage_reached"].astype(str).unique())
        raise ValueError(f"unknown stage values: {bad}")
    at_risk = reached >= k - 1
    table = (
        pd.DataFrame({"group": df.loc[at_risk, attribute], "selected": reached[at_risk] >= k})
        .groupby("group")["selected"]
        .agg(n="size", selected="sum")
    )
    table["selected"] = table["selected"].astype(int)
    table["rate"] = table["selected"] / table["n"]
    return table


def audit_stage(
    df: pd.DataFrame,
    attribute: str,
    stage: str,
    reference: str = "highest",
    min_n: int = schema.MIN_GROUP_N,
) -> pd.DataFrame:
    """Selection rates + AIR + four-fifths flag for one stage transition.

    Columns: n, selected, rate, share, reference_rate, air, eligible, below_four_fifths, note
    * below_four_fifths is True/False only for eligible groups; None otherwise (no verdict).
    """
    t = stage_rates(df, attribute, stage)
    t["share"] = t["n"] / t["n"].sum()
    known = ~t.index.isin(schema.NON_RESPONSE_VALUES)
    t["eligible"] = known & (t["n"] >= min_n)

    if reference == "highest":
        pool = t[t["eligible"]]
        if pool.empty:
            raise ValueError(f"no group has n >= {min_n} at stage '{stage}'")
        t["reference_rate"] = pool["rate"].max()
    elif reference == "pooled":
        tot_n = t.loc[known, "n"].sum()
        tot_sel = t.loc[known, "selected"].sum()
        rest_n = tot_n - t["n"].where(known, 0)
        rest_sel = tot_sel - t["selected"].where(known, 0)
        t["reference_rate"] = rest_sel / rest_n
    elif reference in t.index:
        t["reference_rate"] = t.loc[reference, "rate"]
    else:
        raise ValueError(f"reference must be 'highest', 'pooled', or a group in {list(t.index)}")

    t["air"] = t["rate"] / t["reference_rate"]
    t.loc[~known, "air"] = float("nan")
    t["below_four_fifths"] = (t["air"] < FOUR_FIFTHS).where(t["eligible"], None)

    notes = []
    for g, row in t.iterrows():
        if g in schema.NON_RESPONSE_VALUES:
            notes.append("non-response: reported separately")
        elif row["n"] < min_n:
            notes.append(f"caution: n < {min_n}, no four-fifths verdict")
        elif row["share"] < LL144_MIN_SHARE:
            notes.append("under 2% of data (LL144 may exclude)")
        else:
            notes.append("")
    t["note"] = notes
    t.insert(0, "stage", stage)
    return t.sort_values("air")


def audit_funnel(
    df: pd.DataFrame,
    attribute: str,
    reference: str = "highest",
    min_n: int = schema.MIN_GROUP_N,
) -> pd.DataFrame:
    """audit_stage for every transition (screened, interviewed, offered, hired), stacked."""
    parts = [audit_stage(df, attribute, s, reference, min_n) for s in schema.STAGES[1:]]
    return pd.concat(parts).rename_axis(attribute).reset_index()


if __name__ == "__main__":  # quick look: python -m fairhire.fairhire_stats
    from fairhire.make_demo_data import generate

    demo = generate()
    pd.set_option("display.width", 160)
    for ref in ("highest", "pooled"):
        print(f"\n=== race_ethnicity, interviewed, reference={ref} ===")
        cols = ["n", "selected", "rate", "reference_rate", "air", "below_four_fifths", "note"]
        print(audit_stage(demo, "race_ethnicity", "interviewed", ref)[cols].round(3).to_string())
