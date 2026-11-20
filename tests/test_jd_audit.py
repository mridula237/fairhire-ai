import re

import pytest

from fairhire.jd_audit import (
    audit_jd,
    count_syllables,
    highlighted_html,
    load_flags,
    readability,
    report_markdown,
)
from fairhire.jd_batch import score_folder

BIASED = """We are a young, energetic team looking for a rockstar data analyst who is competitive and driven.
- Bachelor's degree required
- Must be a digital native and a culture fit
- Native English speaker
- Must be able to lift 50 lbs
He will lead our analytics."""

INCLUSIVE = """Program Coordinator
You will support families, collaborate with partner schools, and help children thrive.
- Coordinate weekly sessions and share updates with the team
- Two years of related experience, or an equivalent mix of education and experience
- Fluent in English and Spanish preferred
Pay: $48,000 - $55,000 per year, plus health insurance, dental, and paid time off.
We provide reasonable accommodations to applicants with disabilities."""


def test_flag_yaml_is_valid():
    flags = load_flags()
    ids = [f["id"] for f in flags]
    assert len(ids) == len(set(ids))
    for f in flags:
        assert f["severity"] in {"high", "medium", "low"}
        assert f["why"] and f["suggestion"] and f["patterns"]
        for p in f["patterns"]:
            re.compile(p)


def test_biased_posting_flags_and_score():
    r = audit_jd(BIASED)
    kinds = {h.kind for h in r.flags}
    assert {"young_team", "rockstar", "degree_required", "digital_native", "culture_fit", "native_speaker",
            "lift_requirement", "gendered_pronoun"} <= kinds
    assert r.gender_coding == "strongly masculine-coded"
    assert not r.has_pay_range
    assert r.score <= 40
    # every lost point is itemized
    assert 100 - sum(p for _, p in r.deductions) == r.score or r.score == 0


def test_inclusive_posting_scores_high():
    r = audit_jd(INCLUSIVE)
    assert r.flags == []  # "or equivalent" suppresses the degree flag; no other flags
    assert r.has_pay_range and r.has_benefits
    assert r.gender_coding in {"feminine-coded", "strongly feminine-coded", "neutral"}
    assert r.score >= 90


def test_rewrite_applies_only_safe_replacements():
    r = audit_jd(BIASED)
    assert "rockstar" not in r.rewrite and "skilled data analyst" in r.rewrite
    assert "Must be comfortable with digital tools" in r.rewrite
    assert "culture add" in r.rewrite
    assert "degree or equivalent experience required" in r.rewrite
    assert "He will" in r.rewrite  # needs human judgment: left alone, still flagged


@pytest.mark.parametrize(("word", "n"), [("cat", 1), ("table", 2), ("collaborate", 4), ("opportunity", 5),
                                         ("responsibilities", 6), ("the", 1)])
def test_syllables(word, n):
    assert count_syllables(word) == n


def test_readability_simple_vs_dense():
    simple = readability("You will help kids read. You will plan fun games. We pay well.")
    dense = readability("The incumbent will operationalize cross-functional organizational initiatives, "
                        "facilitating comprehensive stakeholder communication and institutional accountability.")
    assert simple["fk_grade"] < 4 < 14 < dense["fk_grade"]


def test_highlight_escapes_html():
    r = audit_jd("<script>alert(1)</script> We need a rockstar.")
    out = highlighted_html(r)
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<mark" in out


def test_report_markdown_and_batch(tmp_path):
    md = report_markdown(audit_jd(BIASED), "Sample")
    assert "JD Equity Score" in md and "digital native" in md
    (tmp_path / "jd01.txt").write_text(BIASED)
    (tmp_path / "jd02.txt").write_text(INCLUSIVE)
    (tmp_path / "notes.csv").write_text("ignored")
    df, flags = score_folder(tmp_path)
    assert list(df["file"]) == ["jd01.txt", "jd02.txt"]
    assert df.loc[0, "score"] < df.loc[1, "score"]
    assert flags["digital_native"] == 1


def test_young_professionals_board_is_not_age_coded():
    r = audit_jd("Manage the Young Professionals Board and staff its meetings. " * 3)
    assert "young_team" not in {h.kind for h in r.flags}
    r2 = audit_jd("We are hiring young professionals to join our team. " * 3)
    assert "young_team" in {h.kind for h in r2.flags}


def test_benefits_and_eeo_boilerplate_not_gender_coded():
    r = audit_jd("Medical coverage for you and your dependents. Pay depending on experience. "
                 "We welcome individuals with disabilities. We value dependable, independent people.")
    assert "dependents" not in r.feminine and "depending" not in r.feminine
    assert "individuals" not in r.masculine
    assert "dependable" in r.feminine and "independent" in r.masculine  # real uses still count
