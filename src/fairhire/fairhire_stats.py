"""FairHire statistics engine: selection rates, adverse impact ratios, and significance tests.

Definitions (full write-up with citations: docs/methods.md)
-----------
* Selection rate into stage k = applicants who reached stage k / applicants who reached stage k-1.
* Adverse impact ratio (AIR) = group's rate / reference rate.
* Reference options:
    - "highest": highest rate among eligible groups (29 CFR 1607.4(D), NYC LL144). Eligible =
      known response and n >= min_n, so a tiny lucky group can't become the benchmark.
    - "pooled":  each group vs. all OTHER known-response applicants pooled. More stable.
    - any group name, e.g. "White": compare everyone to that group.

Per group vs. its reference:
* two-proportion z-test (pooled SE), or Fisher's exact when any expected cell count < 5
* standard-deviation (Hazelwood) test: (observed - expected) / sqrt(n p (1-p)), p = overall rate
* Wilson 95% CI for the group's rate; Katz log 95% CI for the AIR
Per stage: chi-square test of independence across all eligible groups.

Multiple comparisons: an audit runs many tests (groups x stages), so some p < 0.05 by chance.
Verdicts use Holm-adjusted p-values (p_adjusted) over the family audited: one stage in
audit_stage, the whole funnel in audit_funnel. Raw p_value is kept for transparency.

Verdict (combines practical and statistical significance, as 1607.4(D) does; "p" = p_adjusted):
* "adverse impact indicated":  AIR < 0.80 and p < alpha
* "inconclusive: not significant": AIR < 0.80 but p >= alpha (often small numbers)
* "significant gap above 4/5":  AIR >= 0.80 but p < alpha and rate below reference
* "no adverse impact indicated": otherwise
* "no verdict": non-response, n < min_n, or the group IS the reference
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from fairhire import schema

FOUR_FIFTHS = 0.80
ALPHA = 0.05
Z95 = 1.959963984540054
LL144_MIN_SHARE = 0.02  # NYC LL144: categories under 2% of data may be excluded from impact ratios

_ORDER = {s: i for i, s in enumerate(schema.STAGES)}


# ---------------------------------------------------------------------------
# Building blocks (pure functions; each one is unit-tested against hand calculations)
# ---------------------------------------------------------------------------

def wilson_ci(selected: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = selected / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test with pooled standard error. Returns (z, p)."""
    if n1 == 0 or n2 == 0:
        return (math.nan, math.nan)
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (math.nan, math.nan)
    z = (x1 / n1 - x2 / n2) / se
    return (z, 2 * stats.norm.sf(abs(z)))


def fisher_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided Fisher's exact test on the 2x2 table [[x1, n1-x1], [x2, n2-x2]]."""
    return float(stats.fisher_exact([[x1, n1 - x1], [x2, n2 - x2]]).pvalue)


def min_expected_2x2(x1: int, n1: int, x2: int, n2: int) -> float:
    total, sel = n1 + n2, x1 + x2
    if total == 0:
        return 0.0
    return min(n * s / total for n in (n1, n2) for s in (sel, total - sel))


def sd_test(selected: int, n: int, overall_rate: float) -> float:
    """Hazelwood standard-deviation test: (observed - expected) / binomial SD.
    Negative = fewer selected than expected. |value| > 2 to 3 is the conventional concern line."""
    sd = math.sqrt(n * overall_rate * (1 - overall_rate))
    return math.nan if sd == 0 else (selected - n * overall_rate) / sd


def air_ci_katz(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> tuple[float, float]:
    """Katz log-method CI for a ratio of two independent proportions (x1/n1) / (x2/n2).
    Adds 0.5 to zero counts so the interval is defined."""
    if min(n1, n2) == 0:
        return (math.nan, math.nan)
    if x1 == 0 or x2 == 0:
        x1, x2, n1, n2 = x1 + 0.5, x2 + 0.5, n1 + 0.5, n2 + 0.5
    ratio = (x1 / n1) / (x2 / n2)
    se = math.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
    return (ratio * math.exp(-z * se), ratio * math.exp(z * se))


def chi_square(table: pd.DataFrame) -> tuple[float, float, int]:
    """Chi-square test of independence (group x selected/not). Returns (chi2, p, dof)."""
    if len(table) < 2:
        return (math.nan, math.nan, 0)
    obs = np.column_stack([table["selected"], table["n"] - table["selected"]])
    if (obs.sum(axis=0) == 0).any():
        return (math.nan, math.nan, 0)
    res = stats.chi2_contingency(obs, correction=False)
    return (float(res.statistic), float(res.pvalue), int(res.dof))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN entries are skipped and stay NaN).
    Controls the chance of ANY false alarm across all tests in the family."""
    idx = [i for i, p in enumerate(pvalues) if not math.isnan(p)]
    m = len(idx)
    out = [math.nan] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


# ---------------------------------------------------------------------------
# Audit tables
# ---------------------------------------------------------------------------

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


def _adjust(t: pd.DataFrame) -> None:
    """Holm-adjust p-values over the eligible, non-reference rows of `t` (in place)."""
    family = t["p_value"].where(t["eligible"] & ~t["is_reference"], math.nan)
    t["p_adjusted"] = holm(family.tolist())


def _verdict(row: pd.Series, alpha: float) -> str:
    if not row["eligible"] or row["is_reference"] or math.isnan(row["p_adjusted"]):
        return "no verdict"
    low = row["air"] < FOUR_FIFTHS
    sig = row["p_adjusted"] < alpha
    if low and sig:
        return "adverse impact indicated"
    if low:
        return "inconclusive: not significant"
    if sig and row["rate"] < row["reference_rate"]:
        return "significant gap above 4/5"
    return "no adverse impact indicated"


def audit_stage(
    df: pd.DataFrame,
    attribute: str,
    stage: str,
    reference: str = "highest",
    min_n: int = schema.MIN_GROUP_N,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Full audit table for one stage transition, one row per group.

    Columns: stage, n, selected, rate, rate_ci_low/high, share, eligible, is_reference,
    reference_rate, ref_n, ref_selected, air, air_ci_low/high, below_four_fifths,
    test, p_value, p_adjusted, sd_units, verdict, note.
    Stage-level chi-square results are in table.attrs["chi_square"].
    """
    t = stage_rates(df, attribute, stage)
    t["share"] = t["n"] / t["n"].sum()
    known = ~t.index.isin(schema.NON_RESPONSE_VALUES)
    t["eligible"] = known & (t["n"] >= min_n)

    ci = [wilson_ci(s, n) for s, n in zip(t["selected"], t["n"], strict=True)]
    t["rate_ci_low"], t["rate_ci_high"] = zip(*ci, strict=True)

    # Reference counts per row
    t["is_reference"] = False
    if reference == "highest":
        pool = t[t["eligible"]]
        if pool.empty:
            raise ValueError(f"no group has n >= {min_n} at stage '{stage}'")
        ref = pool["rate"].idxmax()
        t["ref_n"], t["ref_selected"] = t.loc[ref, "n"], t.loc[ref, "selected"]
        t.loc[ref, "is_reference"] = True
    elif reference == "pooled":
        tot_n, tot_sel = t.loc[known, "n"].sum(), t.loc[known, "selected"].sum()
        t["ref_n"] = tot_n - t["n"].where(known, 0)
        t["ref_selected"] = tot_sel - t["selected"].where(known, 0)
    elif reference in t.index:
        t["ref_n"], t["ref_selected"] = t.loc[reference, "n"], t.loc[reference, "selected"]
        t.loc[reference, "is_reference"] = True
    else:
        raise ValueError(f"reference must be 'highest', 'pooled', or a group in {list(t.index)}")
    t["ref_n"] = t["ref_n"].astype(int)
    t["ref_selected"] = t["ref_selected"].astype(int)
    t["reference_rate"] = t["ref_selected"] / t["ref_n"]

    t["air"] = t["rate"] / t["reference_rate"]
    t.loc[~known, "air"] = math.nan
    t["below_four_fifths"] = (t["air"] < FOUR_FIFTHS).where(t["eligible"], None)

    overall_rate = t.loc[known, "selected"].sum() / t.loc[known, "n"].sum()
    tests, pvals, sds, air_lo, air_hi = [], [], [], [], []
    for g, r in t.iterrows():
        x1, n1, x2, n2 = int(r["selected"]), int(r["n"]), int(r["ref_selected"]), int(r["ref_n"])
        if g in schema.NON_RESPONSE_VALUES or r["is_reference"]:
            tests.append("")
            pvals.append(math.nan)
            sds.append(math.nan)
            air_lo.append(math.nan)
            air_hi.append(math.nan)
            continue
        if min_expected_2x2(x1, n1, x2, n2) < 5:
            tests.append("fisher")
            pvals.append(fisher_p(x1, n1, x2, n2))
        else:
            tests.append("z")
            pvals.append(two_proportion_z(x1, n1, x2, n2)[1])
        sds.append(sd_test(x1, n1, overall_rate))
        lo, hi = air_ci_katz(x1, n1, x2, n2)
        air_lo.append(lo)
        air_hi.append(hi)
    t["test"], t["p_value"], t["sd_units"] = tests, pvals, sds
    t["air_ci_low"], t["air_ci_high"] = air_lo, air_hi
    _adjust(t)
    t["verdict"] = [_verdict(r, alpha) for _, r in t.iterrows()]

    notes = []
    for g, row in t.iterrows():
        if g in schema.NON_RESPONSE_VALUES:
            notes.append("non-response: reported separately")
        elif row["is_reference"]:
            notes.append("reference group")
        elif row["n"] < min_n:
            notes.append(f"caution: n < {min_n}, no verdict")
        elif row["share"] < LL144_MIN_SHARE:
            notes.append("under 2% of data (LL144 may exclude)")
        else:
            notes.append("")
    t["note"] = notes
    t.insert(0, "stage", stage)
    t.attrs["chi_square"] = dict(zip(("chi2", "p_value", "dof"), chi_square(t[t["eligible"]]), strict=True))
    t.attrs["reference"] = reference
    return t.sort_values("air")


def audit_funnel(
    df: pd.DataFrame,
    attribute: str,
    reference: str = "highest",
    min_n: int = schema.MIN_GROUP_N,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """audit_stage for every transition, stacked. p-values are Holm-adjusted across the WHOLE
    funnel (every group x stage test), so verdicts here can be stricter than per-stage ones."""
    parts = [audit_stage(df, attribute, s, reference, min_n, alpha) for s in schema.STAGES[1:]]
    full = pd.concat(parts).rename_axis(attribute).reset_index()
    _adjust(full)
    full["verdict"] = [_verdict(r, alpha) for _, r in full.iterrows()]
    return full


SUMMARY_COLUMNS = [
    "stage", "n", "selected", "rate", "reference_rate", "air", "air_ci_low", "air_ci_high",
    "test", "p_value", "p_adjusted", "sd_units", "verdict", "note",
]


if __name__ == "__main__":  # quick look: python -m fairhire.fairhire_stats
    from fairhire.make_demo_data import generate

    demo = generate()
    pd.set_option("display.width", 220)
    for ref in ("highest", "pooled"):
        t = audit_stage(demo, "race_ethnicity", "interviewed", ref)
        print(f"\n=== race_ethnicity, screened -> interviewed, reference={ref} ===")
        print(t[SUMMARY_COLUMNS[1:]].round(3).to_string())
        c = t.attrs["chi_square"]
        print(f"chi-square (eligible groups): chi2={c['chi2']:.2f}, dof={c['dof']}, p={c['p_value']:.2g}")
