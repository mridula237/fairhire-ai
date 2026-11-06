"""Hand-calculated checks for every statistical building block.

Worked example used throughout (screened -> interviewed):
    Group A: 50 of 100 selected (rate 0.50)
    Group B: 30 of 100 selected (rate 0.30)   AIR = 0.30 / 0.50 = 0.60

Hand calculations (shown so a reviewer can redo them on paper):
    pooled p = 80/200 = 0.40
    SE = sqrt(0.40 * 0.60 * (1/100 + 1/100)) = 0.069282
    z  = (0.30 - 0.50) / 0.069282 = -2.88675   two-sided p = 0.0038924
    chi-square (2x2, no continuity correction) = z^2 = 8.3333
    SD test for B: expected = 100 * 0.40 = 40; SD = sqrt(100 * 0.4 * 0.6) = 4.89898
                   (30 - 40) / 4.89898 = -2.0412
    Wilson 95% CI for 30/100 = (0.2189, 0.3958)
    Katz CI for AIR 0.60: SE(log) = sqrt(1/30 - 1/100 + 1/50 - 1/100) = 0.182574
                   0.60 * exp(+/-1.96 * 0.182574) = (0.4195, 0.8582)
    Fisher's exact two-sided p = 0.0059373 (cross-checked by summing hypergeometric terms)
"""

import math

import pandas as pd
import pytest

from fairhire.fairhire_stats import (
    air_ci_katz,
    audit_stage,
    chi_square,
    fisher_p,
    min_expected_2x2,
    sd_test,
    two_proportion_z,
    wilson_ci,
)


def test_two_proportion_z_hand():
    z, p = two_proportion_z(30, 100, 50, 100)
    assert z == pytest.approx(-2.88675, abs=1e-4)
    assert p == pytest.approx(0.0038924, abs=1e-6)


def test_chi_square_equals_z_squared_for_2x2():
    table = pd.DataFrame({"n": [100, 100], "selected": [50, 30]}, index=["A", "B"])
    chi2, p, dof = chi_square(table)
    assert chi2 == pytest.approx(8.3333, abs=1e-3)
    assert dof == 1
    assert p == pytest.approx(0.0038924, abs=1e-6)  # same p as the z-test


def test_sd_test_hand():
    assert sd_test(30, 100, 0.40) == pytest.approx(-2.0412, abs=1e-4)


def test_wilson_hand():
    lo, hi = wilson_ci(30, 100)
    assert lo == pytest.approx(0.2189, abs=1e-4)
    assert hi == pytest.approx(0.3958, abs=1e-4)
    assert wilson_ci(0, 10)[0] == 0.0  # stays inside [0, 1]


def test_katz_air_ci_hand():
    lo, hi = air_ci_katz(30, 100, 50, 100)
    assert lo == pytest.approx(0.4195, abs=1e-3)
    assert hi == pytest.approx(0.8582, abs=1e-3)


def test_fisher_hand():
    assert fisher_p(30, 100, 50, 100) == pytest.approx(0.0059373, abs=1e-6)


def test_fisher_used_when_expected_counts_small():
    # 2/10 vs 9/10: smallest expected cell = 10 * 9 / 20 = 4.5 < 5
    assert min_expected_2x2(2, 10, 9, 10) == pytest.approx(4.5)
    rows = (
        [{"race_ethnicity": "S", "stage_reached": "interviewed"}] * 2
        + [{"race_ethnicity": "S", "stage_reached": "screened"}] * 8
        + [{"race_ethnicity": "R", "stage_reached": "interviewed"}] * 9
        + [{"race_ethnicity": "R", "stage_reached": "screened"}] * 1
    )
    t = audit_stage(pd.DataFrame(rows), "race_ethnicity", "interviewed", reference="R", min_n=0)
    assert t.loc["S", "test"] == "fisher"
    assert t.loc["S", "p_value"] == pytest.approx(0.0054775, abs=1e-6)


def _df(groups: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """groups = {name: (selected, n)} at the screened -> interviewed step."""
    rows = []
    for g, (sel, n) in groups.items():
        rows += [{"race_ethnicity": g, "stage_reached": "interviewed"}] * sel
        rows += [{"race_ethnicity": g, "stage_reached": "screened"}] * (n - sel)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("groups", "verdict"),
    [
        ({"Ref": (50, 100), "G": (30, 100)}, "adverse impact indicated"),          # AIR .60, p=.004
        ({"Ref": (50, 100), "G": (14, 40)}, "inconclusive: not significant"),      # AIR .70, p~.11
        ({"Ref": (5000, 10000), "G": (4200, 10000)}, "significant gap above 4/5"),  # AIR .84, huge n
        ({"Ref": (50, 100), "G": (48, 100)}, "no adverse impact indicated"),       # AIR .96, p~.78
    ],
)
def test_verdicts(groups, verdict):
    t = audit_stage(_df(groups), "race_ethnicity", "interviewed", reference="Ref")
    assert t.loc["G", "verdict"] == verdict
    assert t.loc["Ref", "verdict"] == "no verdict"  # a group is never compared to itself


def test_full_row_matches_hand_example():
    t = audit_stage(_df({"A": (50, 100), "B": (30, 100)}), "race_ethnicity", "interviewed", reference="A")
    b = t.loc["B"]
    assert b["air"] == pytest.approx(0.60)
    assert b["test"] == "z"
    assert b["p_value"] == pytest.approx(0.0038924, abs=1e-6)
    assert b["sd_units"] == pytest.approx(-2.0412, abs=1e-4)
    assert (b["air_ci_low"], b["air_ci_high"]) == pytest.approx((0.4195, 0.8582), abs=1e-3)
    assert t.attrs["chi_square"]["chi2"] == pytest.approx(8.3333, abs=1e-3)
    assert math.isnan(t.loc["A", "p_value"])


def test_holm_hand():
    from fairhire.fairhire_stats import holm

    # sorted p: .01, .02, .04 with m = 3 -> .03, max(.03, .04) = .04, max(.04, .04) = .04
    assert holm([0.04, 0.01, math.nan, 0.02]) == pytest.approx([0.04, 0.03, math.nan, 0.04], nan_ok=True)


def test_no_false_alarms_on_demo_funnel():
    """Only the injected gap (Hispanic or Latino, interview stage) should be flagged."""
    from fairhire.fairhire_stats import add_intersection, audit_funnel
    from fairhire.make_demo_data import GeneratorConfig, generate

    df = add_intersection(generate(GeneratorConfig(n=5000, seed=42)))
    for attr in ("race_ethnicity", "gender_x_race_ethnicity"):
        f = audit_funnel(df, attr, reference="pooled")
        flagged = f[f["verdict"].isin(["adverse impact indicated", "significant gap above 4/5"])]
        assert set(flagged["stage"]) <= {"interviewed"}, flagged[[attr, "stage", "p_adjusted"]]
        assert all("Hispanic or Latino" in g for g in flagged[attr])
