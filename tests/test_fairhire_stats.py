import math

import pandas as pd
import pytest

from fairhire.fairhire_stats import add_intersection, audit_funnel, audit_stage, stage_rates
from fairhire.make_demo_data import GeneratorConfig, generate


def _rows(group: str, stages: dict[str, int]) -> list[dict]:
    """stages = {'applied': 10, 'screened': 5, ...}: count who STOPPED at each stage."""
    return [
        {"race_ethnicity": group, "gender": "Female", "stage_reached": s}
        for s, k in stages.items()
        for _ in range(k)
    ]


@pytest.fixture
def hand() -> pd.DataFrame:
    """Hand-calculable example at the screened -> interviewed step.

    A: 100 screened, 50 interviewed  -> rate 0.50
    B: 100 screened, 30 interviewed  -> rate 0.30   AIR vs A = 0.60
    C:  10 screened,  9 interviewed  -> rate 0.90   (tiny lucky group, n < 30)
    Unknown: 40 screened, 10 interviewed
    """
    rows = (
        _rows("A", {"screened": 50, "interviewed": 50})
        + _rows("B", {"screened": 70, "interviewed": 30})
        + _rows("C", {"screened": 1, "interviewed": 9})
        + _rows("Unknown", {"screened": 30, "interviewed": 10})
        + _rows("A", {"applied": 20})  # never screened: must not count at this stage
    )
    return pd.DataFrame(rows)


def test_stage_rates_hand(hand):
    t = stage_rates(hand, "race_ethnicity", "interviewed")
    assert t.loc["A", "n"] == 100 and t.loc["A", "selected"] == 50
    assert t.loc["B", "rate"] == pytest.approx(0.30)
    assert t.loc["C", "rate"] == pytest.approx(0.90)


def test_highest_reference_ignores_small_and_unknown_groups(hand):
    t = audit_stage(hand, "race_ethnicity", "interviewed", reference="highest")
    assert t.loc["A", "reference_rate"] == pytest.approx(0.50)  # not C's lucky 0.90
    assert t.loc["B", "air"] == pytest.approx(0.60)
    assert t.loc["B", "below_four_fifths"] == True  # noqa: E712
    assert t.loc["C", "below_four_fifths"] is None  # n < 30: no verdict
    assert math.isnan(t.loc["Unknown", "air"])


def test_pooled_reference_hand(hand):
    t = audit_stage(hand, "race_ethnicity", "interviewed", reference="pooled")
    # B vs rest of known (A + C): (50 + 9) / (100 + 10) = 0.53636...
    assert t.loc["B", "reference_rate"] == pytest.approx(59 / 110)
    assert t.loc["B", "air"] == pytest.approx(0.30 / (59 / 110))


def test_named_reference(hand):
    t = audit_stage(hand, "race_ethnicity", "interviewed", reference="B")
    assert t.loc["A", "air"] == pytest.approx(0.50 / 0.30)
    with pytest.raises(ValueError):
        audit_stage(hand, "race_ethnicity", "interviewed", reference="Z")


def test_invalid_stage(hand):
    with pytest.raises(ValueError):
        stage_rates(hand, "race_ethnicity", "applied")


def test_recovers_injected_disparity_on_demo():
    df = generate(GeneratorConfig(n=60000, seed=1))
    t = audit_stage(df, "race_ethnicity", "interviewed", reference="pooled")
    assert t.loc["Hispanic or Latino", "air"] == pytest.approx(0.65, abs=0.04)
    assert t.loc["Hispanic or Latino", "below_four_fifths"] == True  # noqa: E712
    # groups with no injected gap are not flagged
    for g in ["White", "Black or African American", "Asian"]:
        assert t.loc[g, "below_four_fifths"] == False  # noqa: E712


def test_funnel_and_intersection_shapes():
    df = add_intersection(generate(GeneratorConfig(n=5000, seed=42)))
    f = audit_funnel(df, "race_ethnicity")
    assert set(f["stage"]) == {"screened", "interviewed", "offered", "hired"}
    x = audit_stage(df, "gender_x_race_ethnicity", "screened")
    assert "Female | Hispanic or Latino" in x.index
    assert "Unknown" in x.index
