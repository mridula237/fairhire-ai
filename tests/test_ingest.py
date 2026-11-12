import pandas as pd
import pytest

from fairhire import schema
from fairhire.fairhire_stats import audit_stage
from fairhire.ingest import (
    apply_mapping,
    guess_stage,
    identifier_columns,
    stage_from_flags,
    suggest_mapping,
    suggest_stage_flags,
    unmapped_stage_values,
    validate,
)


@pytest.fixture
def messy() -> pd.DataFrame:
    """Looks like a real ATS export: odd headers, free-text stages, short codes, blanks."""
    return pd.DataFrame(
        {
            "Candidate ID": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "Requisition": ["R-9", "R-9", "R-9", "R-10", "R-10", "R-10"],
            "Job Title": ["Analyst"] * 3 + ["Teacher"] * 3,
            "Status": ["Application Received", "Phone Screen", "Onsite", "Offer Extended", "Hired",
                       "Talent Pool"],
            "Gender": ["F", "M", "Declined", "female", None, "Woman"],
            "Race": ["Hispanic", "White", "", "Black", "Asian", "Latinx"],
            "Email": ["a@x.com"] * 6,
        }
    )


def test_suggest_mapping_messy(messy):
    m = suggest_mapping(list(messy.columns))
    assert m["applicant_id"] == "Candidate ID"
    assert m["req_id"] == "Requisition"
    assert m["job_title"] == "Job Title"
    assert m["stage_reached"] == "Status"
    assert m["gender"] == "Gender"
    assert m["race_ethnicity"] == "Race"
    assert m["source"] is None


def test_identifier_columns():
    cols = ["Candidate ID", "Email", "Full Name", "Phone", "Home Address", "DOB", "Date of Birth", "Job Name",
            "Gender"]
    assert identifier_columns(cols) == ["Email", "Full Name", "Phone", "Home Address", "DOB", "Date of Birth",
                                        "Job Name"]
    # a column the user mapped to a canonical field is allowed
    assert "Job Name" not in identifier_columns(cols, mapped={"Job Name"})


@pytest.mark.parametrize(
    ("raw", "stage"),
    [("Application Received", "applied"), ("Phone Screen", "screened"), ("Onsite", "interviewed"),
     ("Hiring Manager Interview", "interviewed"), ("Offer Extended", "offered"), ("HIRED", "hired"),
     ("Talent Pool", None)],
)
def test_guess_stage(raw, stage):
    assert guess_stage(raw) == stage


def test_apply_mapping_normalizes_and_validate_blocks_email(messy):
    m = suggest_mapping(list(messy.columns))
    assert unmapped_stage_values(messy, "Status") == ["Talent Pool"]
    df = apply_mapping(messy, m, stage_value_map={"Talent Pool": "applied"})
    assert list(df.columns) == schema.ALL_COLUMNS
    assert df["stage_reached"].tolist() == ["applied", "screened", "interviewed", "offered", "hired", "applied"]
    assert df["gender"].tolist() == ["Female", "Male", "Prefer not to say", "Female", "Unknown", "Female"]
    assert df["race_ethnicity"].tolist() == ["Hispanic or Latino", "White", "Unknown",
                                             "Black or African American", "Asian", "Hispanic or Latino"]
    rep = validate(df, list(messy.columns), m)
    assert not rep.ok
    assert any("'Email'" in e for e in rep.errors)

    clean = messy.drop(columns="Email")
    rep2 = validate(apply_mapping(clean, m, stage_value_map={"Talent Pool": "applied"}), list(clean.columns), m)
    assert rep2.ok, rep2.errors
    assert any("Only 6 rows" in w for w in rep2.warnings)


def test_unplaced_stage_is_an_error(messy):
    clean = messy.drop(columns="Email")
    m = suggest_mapping(list(clean.columns))
    rep = validate(apply_mapping(clean, m), list(clean.columns), m)  # Talent Pool not mapped
    assert any("couldn't place" in e for e in rep.errors)


def test_stage_from_yes_no_columns():
    raw = pd.DataFrame(
        {
            "Applicant": ["a", "b", "c", "d"],
            "Passed Screen": ["Yes", "Yes", "No", "yes"],
            "Interviewed": [1, 0, 0, 1],
            "Offer": ["Y", "", "", "N"],
            "Hired": [True, False, False, False],
            "Sex": ["F", "M", "F", "M"],
        }
    )
    flags = suggest_stage_flags(list(raw.columns))
    assert flags == {"screened": "Passed Screen", "interviewed": "Interviewed", "offered": "Offer",
                     "hired": "Hired"}
    assert stage_from_flags(raw, flags).tolist() == ["hired", "screened", "applied", "interviewed"]
    m = suggest_mapping(list(raw.columns))
    df = apply_mapping(raw, m, stage_flags=flags)
    rep = validate(df, list(raw.columns), m, stage_flags=flags)
    assert rep.ok, rep.errors


def test_demo_file_round_trips_through_ingest():
    """The canonical template maps to itself and audits identically."""
    from fairhire.make_demo_data import GeneratorConfig, generate

    demo = generate(GeneratorConfig(n=3000, seed=5))
    m = suggest_mapping(list(demo.columns))
    assert all(m[f] == f for f in schema.ALL_COLUMNS)
    df = apply_mapping(demo, m)
    assert validate(df, list(demo.columns), m).ok
    a = audit_stage(demo, "race_ethnicity", "interviewed")
    b = audit_stage(df, "race_ethnicity", "interviewed")
    pd.testing.assert_series_equal(a["air"], b["air"])
