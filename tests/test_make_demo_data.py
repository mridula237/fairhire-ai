import pandas as pd
import pytest

from fairhire import schema
from fairhire.make_demo_data import (
    Disparity,
    GeneratorConfig,
    air_vs_rest,
    generate,
    ground_truth,
)


@pytest.fixture(scope="module")
def demo() -> pd.DataFrame:
    return generate(GeneratorConfig(n=5000, seed=42))


def test_columns_match_schema(demo):
    assert list(demo.columns) == schema.ALL_COLUMNS


def test_row_count_and_unique_ids(demo):
    assert len(demo) == 5000
    assert demo["applicant_id"].is_unique


def test_categories_are_valid(demo):
    assert set(demo["stage_reached"]) <= set(schema.STAGES)
    assert set(demo["gender"]) <= set(schema.GENDER_VALUES)
    assert set(demo["race_ethnicity"]) <= set(schema.RACE_ETHNICITY_VALUES)
    assert set(demo["age_band"]) <= set(schema.AGE_BAND_VALUES)
    assert set(demo["education_level"]) <= set(schema.EDUCATION_VALUES)


def test_no_identifier_columns(demo):
    for col in demo.columns:
        if col == "applicant_id":
            continue
        assert not any(p in col.lower() for p in schema.IDENTIFIER_PATTERNS), col


def test_reproducible_with_seed():
    a = generate(GeneratorConfig(n=500, seed=7))
    b = generate(GeneratorConfig(n=500, seed=7))
    pd.testing.assert_frame_equal(a, b)


def test_funnel_narrows(demo):
    order = {s: i for i, s in enumerate(schema.STAGES)}
    reached = demo["stage_reached"].map(order)
    counts = [(reached >= i).sum() for i in range(len(schema.STAGES))]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] > 100  # enough hires for Module 3 later


def test_default_injected_disparity_recovered(demo):
    """Week 1 benchmark: interview-stage AIR of about 0.65 for one group."""
    air, n = air_vs_rest(demo, "race_ethnicity", "Hispanic or Latino", "interviewed")
    assert n >= schema.MIN_GROUP_N
    assert air == pytest.approx(0.65, abs=0.10)


def test_injected_disparity_tight_at_large_n():
    df = generate(GeneratorConfig(n=60000, seed=1))
    air, _ = air_vs_rest(df, "race_ethnicity", "Hispanic or Latino", "interviewed")
    assert air == pytest.approx(0.65, abs=0.04)


def test_no_disparity_means_air_near_one():
    df = generate(GeneratorConfig(n=60000, seed=3, disparities=[]))
    for group in ["Hispanic or Latino", "Black or African American", "White", "Asian"]:
        air, _ = air_vs_rest(df, "race_ethnicity", group, "interviewed")
        assert air == pytest.approx(1.0, abs=0.06), group


def test_multiple_disparities_and_ground_truth():
    cfg = GeneratorConfig(
        n=60000,
        seed=11,
        disparities=[
            Disparity("race_ethnicity", "Black or African American", "screened", 0.75),
            Disparity("gender", "Female", "offered", 0.70),
        ],
    )
    df = generate(cfg)
    truth = ground_truth(cfg, df)
    for d in truth["injected_disparities"]:
        assert d["realized_air_vs_rest"] == pytest.approx(d["expected_air"], abs=0.05)


def test_disparity_parse_and_validation():
    d = Disparity.parse("gender|Female|offered|0.7")
    assert d == Disparity("gender", "Female", "offered", 0.7)
    with pytest.raises(ValueError):
        Disparity("zip_code", "60601", "offered", 0.7)  # not an audit attribute
    with pytest.raises(ValueError):
        Disparity("gender", "Female", "applied", 0.7)  # nobody "passes into" applied
