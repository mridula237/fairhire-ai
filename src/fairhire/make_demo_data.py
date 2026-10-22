"""Synthetic hiring data generator with injected, tunable stage-level disparities.

Each applicant moves through applied -> screened -> interviewed -> offered -> hired.
At each transition the pass probability is

    p = base_rate[stage] * qualification_factor * group_multiplier

* qualification_factor comes from education and years of experience and is drawn
  from the SAME distribution for every group, so it adds realism without
  confounding the injected gaps.
* group_multiplier defaults to 1.0. A Disparity sets it below 1.0 for one group at
  one stage. Because every other group sits at 1.0, the expected adverse impact
  ratio (AIR) at that stage is approximately the multiplier itself.

The generator also writes a ground-truth JSON next to the CSV so the statistics
engine (week 2) can be tested against known answers.

Usage:
    python -m fairhire.make_demo_data                      # defaults: 5,000 rows
    python -m fairhire.make_demo_data --n 20000 --seed 7 \
        --disparity "race_ethnicity|Hispanic or Latino|interviewed|0.65" \
        --disparity "gender|Female|offered|0.75"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from fairhire import schema

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_RATES = {  # probability of passing INTO this stage, given the previous one
    "screened": 0.45,
    "interviewed": 0.40,
    "offered": 0.35,
    "hired": 0.80,
}

RACE_SHARES = {
    "White": 0.44,
    "Hispanic or Latino": 0.20,
    "Black or African American": 0.16,
    "Asian": 0.10,
    "Two or More Races": 0.03,
    "Native Hawaiian or Other Pacific Islander": 0.01,
    "American Indian or Alaska Native": 0.01,
    "Unknown": 0.05,
}

GENDER_SHARES = {
    "Female": 0.48,
    "Male": 0.46,
    "Nonbinary": 0.02,
    "Prefer not to say": 0.02,
    "Unknown": 0.02,
}

AGE_SHARES = {"Under 40": 0.68, "40 and over": 0.29, "Unknown": 0.03}
SES_SHARES = {"First-generation college": 0.33, "Not first-generation": 0.60, "Unknown": 0.07}
SOURCE_SHARES = {"Job board": 0.45, "Company website": 0.20, "Referral": 0.15, "Campus": 0.12, "Agency": 0.08}
EDU_SHARES = {"High school": 0.15, "Associate": 0.15, "Bachelor": 0.45, "Master": 0.20, "Doctorate": 0.05}
EDU_FACTOR = {"High school": 0.85, "Associate": 0.92, "Bachelor": 1.0, "Master": 1.08, "Doctorate": 1.12}

REQUISITIONS = [
    ("REQ-101", "Program Coordinator"),
    ("REQ-102", "Data Analyst"),
    ("REQ-103", "Administrative Assistant"),
    ("REQ-104", "Teacher, Middle School Math"),
    ("REQ-105", "Customer Success Associate"),
    ("REQ-106", "Software Engineer"),
    ("REQ-107", "Operations Manager"),
    ("REQ-108", "Case Manager"),
    ("REQ-109", "Marketing Specialist"),
    ("REQ-110", "Warehouse Supervisor"),
]


@dataclass(frozen=True)
class Disparity:
    attribute: str      # e.g. "race_ethnicity"
    group: str          # e.g. "Hispanic or Latino"
    stage: str          # destination stage, e.g. "interviewed" (= screened -> interviewed)
    multiplier: float   # 0.65 means the group passes at 65% of everyone else's rate

    def __post_init__(self) -> None:
        if self.attribute not in schema.AUDIT_ATTRIBUTES:
            raise ValueError(f"attribute must be one of {schema.AUDIT_ATTRIBUTES}")
        if self.stage not in BASE_RATES:
            raise ValueError(f"stage must be one of {list(BASE_RATES)}")
        if not 0 < self.multiplier <= 1.5:
            raise ValueError("multiplier must be in (0, 1.5]")

    @classmethod
    def parse(cls, text: str) -> Disparity:
        attribute, group, stage, mult = (p.strip() for p in text.split("|"))
        return cls(attribute, group, stage, float(mult))


DEFAULT_DISPARITIES = [Disparity("race_ethnicity", "Hispanic or Latino", "interviewed", 0.65)]


@dataclass
class GeneratorConfig:
    n: int = 5000
    seed: int = 42
    start_date: date = date(2025, 1, 1)
    end_date: date = date(2026, 9, 30)
    disparities: list[Disparity] = field(default_factory=lambda: list(DEFAULT_DISPARITIES))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _draw(rng: np.random.Generator, shares: dict[str, float], n: int) -> np.ndarray:
    keys = list(shares)
    probs = np.array([shares[k] for k in keys], dtype=float)
    return rng.choice(keys, size=n, p=probs / probs.sum())


def generate(config: GeneratorConfig | None = None) -> pd.DataFrame:
    """Return an applicant-level DataFrame in the canonical FairHire schema."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n

    req_idx = rng.integers(0, len(REQUISITIONS), size=n)
    days = (cfg.end_date - cfg.start_date).days
    app_dates = [cfg.start_date + timedelta(days=int(d)) for d in rng.integers(0, days + 1, size=n)]

    df = pd.DataFrame(
        {
            "applicant_id": [f"A{i:06d}" for i in rng.permutation(n) + 1],
            "req_id": [REQUISITIONS[i][0] for i in req_idx],
            "job_title": [REQUISITIONS[i][1] for i in req_idx],
            "gender": _draw(rng, GENDER_SHARES, n),
            "race_ethnicity": _draw(rng, RACE_SHARES, n),
            "age_band": _draw(rng, AGE_SHARES, n),
            "ses_indicator": _draw(rng, SES_SHARES, n),
            "source": _draw(rng, SOURCE_SHARES, n),
            "education_level": _draw(rng, EDU_SHARES, n),
            "years_experience": np.clip(rng.gamma(shape=2.0, scale=3.0, size=n), 0, 40).round(1),
            "application_date": pd.to_datetime(app_dates).date,
        }
    )

    # Qualification factor: identical distribution across groups (no confounding).
    exp_factor = 0.9 + 0.2 * np.clip(df["years_experience"].to_numpy() / 10.0, 0, 1)
    qual = df["education_level"].map(EDU_FACTOR).to_numpy() * exp_factor

    # Group multipliers per destination stage.
    mult = {stage: np.ones(n) for stage in BASE_RATES}
    for d in cfg.disparities:
        mask = (df[d.attribute] == d.group).to_numpy()
        mult[d.stage][mask] *= d.multiplier

    reached = np.zeros(n, dtype=int)  # index into schema.STAGES; 0 = applied
    alive = np.ones(n, dtype=bool)
    for k, stage in enumerate(schema.STAGES[1:], start=1):
        p = np.clip(BASE_RATES[stage] * qual * mult[stage], 0, 0.98)
        passed = alive & (rng.random(n) < p)
        reached[passed] = k
        alive = passed

    df.insert(3, "stage_reached", [schema.STAGES[i] for i in reached])
    return df[schema.ALL_COLUMNS]


# ---------------------------------------------------------------------------
# Quick check (NOT the week-2 stats engine; just enough to verify ground truth)
# ---------------------------------------------------------------------------

def stage_air(df: pd.DataFrame, attribute: str, stage: str) -> pd.DataFrame:
    """Selection rate into `stage` among those who reached the previous stage,
    and AIR versus the highest-rate group (excluding non-response values)."""
    order = {s: i for i, s in enumerate(schema.STAGES)}
    k = order[stage]
    at_risk = df[df["stage_reached"].map(order) >= k - 1]
    passed = at_risk["stage_reached"].map(order) >= k
    out = (
        pd.DataFrame({"group": at_risk[attribute], "passed": passed})
        .groupby("group")["passed"].agg(n="size", selected="sum")
    )
    out["rate"] = out["selected"] / out["n"]
    eligible = out[(out["n"] >= schema.MIN_GROUP_N) & ~out.index.isin(schema.NON_RESPONSE_VALUES)]
    out["air"] = out["rate"] / eligible["rate"].max()
    return out.sort_values("air")


def air_vs_rest(df: pd.DataFrame, attribute: str, group: str, stage: str) -> tuple[float, int]:
    """AIR of `group` versus all other known-response applicants pooled.

    This is the ground-truth comparison: the injected multiplier is relative to
    everyone else, so pooling the rest removes the small-group noise that the
    'highest-selected group' reference picks up.
    """
    order = {s: i for i, s in enumerate(schema.STAGES)}
    k = order[stage]
    at_risk = df[df["stage_reached"].map(order) >= k - 1]
    passed = at_risk["stage_reached"].map(order) >= k
    in_group = at_risk[attribute] == group
    rest = ~in_group & ~at_risk[attribute].isin(schema.NON_RESPONSE_VALUES)
    return float(passed[in_group].mean() / passed[rest].mean()), int(in_group.sum())


def ground_truth(cfg: GeneratorConfig, df: pd.DataFrame) -> dict:
    injected = []
    for d in cfg.disparities:
        air, n_at_stage = air_vs_rest(df, d.attribute, d.group, d.stage)
        hi = stage_air(df, d.attribute, d.stage)
        injected.append(
            {
                **asdict(d),
                "expected_air": d.multiplier,
                "realized_air_vs_rest": round(air, 3),
                "realized_air_vs_highest": round(float(hi.loc[d.group, "air"]), 3),
                "group_n_at_stage": n_at_stage,
            }
        )
    return {
        "n": cfg.n,
        "seed": cfg.seed,
        "base_rates": BASE_RATES,
        "injected_disparities": injected,
        "note": "All other group-by-stage multipliers are 1.0 (no injected disparity).",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/demo_hiring_data.csv"))
    ap.add_argument(
        "--disparity", action="append", default=None,
        help='"attribute|group|stage|multiplier"; repeatable. Default: Hispanic or Latino at interview, 0.65',
    )
    ap.add_argument("--none", action="store_true", help="generate with no injected disparities")
    args = ap.parse_args(argv)

    if args.none:
        disparities: list[Disparity] = []
    elif args.disparity:
        disparities = [Disparity.parse(t) for t in args.disparity]
    else:
        disparities = list(DEFAULT_DISPARITIES)

    cfg = GeneratorConfig(n=args.n, seed=args.seed, disparities=disparities)
    df = generate(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    truth = ground_truth(cfg, df)
    truth_path = args.out.with_suffix(".ground_truth.json")
    truth_path.write_text(json.dumps(truth, indent=2))

    print(f"Wrote {len(df):,} rows -> {args.out}")
    print(f"Ground truth -> {truth_path}")
    print(df["stage_reached"].value_counts().reindex(schema.STAGES).to_string())
    for d in truth["injected_disparities"]:
        print(f"Injected: {d['group']} at {d['stage']}: expected AIR {d['expected_air']}, "
              f"realized {d['realized_air_vs_rest']} vs rest / "
              f"{d['realized_air_vs_highest']} vs highest group (n={d['group_n_at_stage']})")


if __name__ == "__main__":
    main()
