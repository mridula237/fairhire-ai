"""Turn a messy applicant-tracking-system export into the canonical FairHire schema.

Three steps, all pure functions (the Streamlit app only adds widgets on top):
1. suggest_mapping(columns)            -> {canonical field: uploaded column or None}
2. apply_mapping(raw, mapping, ...)    -> DataFrame with canonical columns, values normalized
3. validate(df, raw_columns, mapping)  -> Report(errors, warnings, info)

Stage can arrive two ways:
* one column with the furthest stage reached ("Phone Screen", "Onsite", "Hired", ...)
* one yes/no column per stage (screened=1, interviewed=0, ...)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from fairhire import schema

NONE = None  # "not in my file"

# ---------------------------------------------------------------------------
# Column-name suggestions
# ---------------------------------------------------------------------------

COLUMN_SYNONYMS: dict[str, list[str]] = {
    "applicant_id": ["applicant_id", "candidate_id", "applicant", "candidate", "app_id", "id"],
    "req_id": ["req_id", "requisition_id", "requisition", "req", "job_id", "posting_id", "job_req"],
    "job_title": ["job_title", "title", "position", "role", "job", "job_name", "position_title"],
    "stage_reached": ["stage_reached", "stage", "status", "furthest_stage", "current_stage",
                      "disposition_stage", "application_status", "step"],
    "gender": ["gender", "sex", "gender_identity"],
    "race_ethnicity": ["race_ethnicity", "race", "ethnicity", "race_and_ethnicity", "eeo_race"],
    "age_band": ["age_band", "age_group", "age_40_plus", "age_range"],
    "ses_indicator": ["ses_indicator", "ses", "first_gen", "first_generation", "pell", "income_tier"],
    "source": ["source", "applicant_source", "candidate_source", "channel", "referral_source"],
    "education_level": ["education_level", "education", "degree", "highest_degree"],
    "years_experience": ["years_experience", "experience", "yoe", "years_of_experience"],
    "application_date": ["application_date", "applied_date", "date_applied", "applied_on", "date"],
}

# One yes/no column per stage
STAGE_FLAG_SYNONYMS: dict[str, list[str]] = {
    "screened": ["screened", "passed_screen", "phone_screen", "screen"],
    "interviewed": ["interviewed", "interview", "onsite", "interviewed_flag"],
    "offered": ["offered", "offer", "offer_extended"],
    "hired": ["hired", "hire", "offer_accepted", "started"],
}


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _tokens(text: str) -> set[str]:
    return set(_key(text).split("_")) - {""}


def suggest_mapping(columns: list[str]) -> dict[str, str | None]:
    """Best-guess uploaded column for each canonical field (exact synonym match, first wins)."""
    by_key = {_key(c): c for c in columns}
    used: set[str] = set()
    out: dict[str, str | None] = {}
    for fld, syns in COLUMN_SYNONYMS.items():
        out[fld] = NONE
        for s in syns:
            col = by_key.get(s)
            if col and col not in used:
                out[fld] = col
                used.add(col)
                break
    return out


def suggest_stage_flags(columns: list[str]) -> dict[str, str | None]:
    by_key = {_key(c): c for c in columns}
    return {
        st: next((by_key[s] for s in syns if s in by_key), NONE)
        for st, syns in STAGE_FLAG_SYNONYMS.items()
    }


def identifier_columns(columns: list[str], mapped: set[str] | None = None) -> list[str]:
    """Columns that look like personal identifiers (name, email, phone, address, SSN, DOB).
    Columns the user mapped to a canonical field (e.g. 'Job Name' -> job_title) are exempt."""
    mapped = mapped or set()
    hits = []
    for c in columns:
        if c in mapped:
            continue
        toks, key = _tokens(c), _key(c)
        if any(p in toks or (len(p) > 4 and p in key) for p in schema.IDENTIFIER_PATTERNS):
            hits.append(c)
    return hits


# ---------------------------------------------------------------------------
# Value normalization
# ---------------------------------------------------------------------------

STAGE_VALUE_SYNONYMS: dict[str, list[str]] = {
    "applied": ["applied", "application", "new", "applicant", "received", "submitted", "rejected_at_application"],
    "screened": ["screened", "screen", "phone_screen", "recruiter_screen", "passed_screen", "assessment"],
    "interviewed": ["interviewed", "interview", "onsite", "on_site", "final_interview", "panel"],
    "offered": ["offered", "offer", "offer_extended", "offer_declined"],
    "hired": ["hired", "hire", "offer_accepted", "accepted", "started", "onboarding"],
}

GENDER_SYNONYMS: dict[str, list[str]] = {
    "Female": ["female", "f", "woman", "women"],
    "Male": ["male", "m", "man", "men"],
    "Nonbinary": ["nonbinary", "non_binary", "nb", "x"],
    "Prefer not to say": ["prefer_not_to_say", "declined", "decline_to_state", "i_don_t_wish_to_answer",
                          "do_not_wish_to_disclose"],
}

RACE_SYNONYMS: dict[str, list[str]] = {
    "Hispanic or Latino": ["hispanic_or_latino", "hispanic", "latino", "latina", "latinx", "hispanic_latino"],
    "White": ["white", "caucasian"],
    "Black or African American": ["black_or_african_american", "black", "african_american"],
    "Asian": ["asian"],
    "Native Hawaiian or Other Pacific Islander": ["native_hawaiian_or_other_pacific_islander",
                                                  "native_hawaiian", "pacific_islander", "nhpi"],
    "American Indian or Alaska Native": ["american_indian_or_alaska_native", "american_indian",
                                         "alaska_native", "native_american", "aian"],
    "Two or More Races": ["two_or_more_races", "two_or_more", "multiracial", "multi_racial", "mixed"],
    "Prefer not to say": ["prefer_not_to_say", "declined", "decline_to_state", "i_don_t_wish_to_answer"],
}

AGE_SYNONYMS: dict[str, list[str]] = {
    "Under 40": ["under_40", "under40", "lt_40", "less_than_40", "no", "0", "false"],
    "40 and over": ["40_and_over", "40", "40_plus", "over_40", "yes", "1", "true"],
}

BLANKS = {"", "nan", "none", "null", "n_a", "na", "unknown", "not_specified", "not_provided"}


def _lookup(table: dict[str, list[str]]) -> dict[str, str]:
    return {s: canon for canon, syns in table.items() for s in syns}


_STAGE = _lookup(STAGE_VALUE_SYNONYMS)
_GENDER = _lookup(GENDER_SYNONYMS)
_RACE = _lookup(RACE_SYNONYMS)
_AGE = _lookup(AGE_SYNONYMS)


def guess_stage(value) -> str | None:
    k = _key(value)
    if k in _STAGE:
        return _STAGE[k]
    for canon, syns in STAGE_VALUE_SYNONYMS.items():  # e.g. "Hiring Manager Interview"
        if any(s in k for s in syns if len(s) > 4):
            return canon
    return None


def normalize_category(series: pd.Series, table: dict[str, str]) -> pd.Series:
    """Map known spellings to canonical labels; blanks -> 'Unknown'; others kept as given."""
    def one(v):
        k = _key(v) if pd.notna(v) else ""
        if k in BLANKS:
            return "Unknown"
        return table.get(k, str(v).strip())
    return series.map(one)


_TRUTHY = {"1", "y", "yes", "true", "t", "x", "pass", "passed"}


def stage_from_flags(raw: pd.DataFrame, flags: dict[str, str | None]) -> pd.Series:
    """Furthest stage from yes/no columns. A later 'yes' implies the earlier stages."""
    reached = pd.Series(0, index=raw.index)
    for k, st in enumerate(schema.STAGES[1:], start=1):
        col = flags.get(st)
        if col:
            yes = raw[col].map(lambda v: _key(v) in _TRUTHY if pd.notna(v) else False)
            reached = reached.where(~yes, k)
    return reached.map(lambda i: schema.STAGES[i])


# ---------------------------------------------------------------------------
# Apply + validate
# ---------------------------------------------------------------------------

@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def apply_mapping(
    raw: pd.DataFrame,
    mapping: dict[str, str | None],
    stage_flags: dict[str, str | None] | None = None,
    stage_value_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the canonical DataFrame. Unmapped optional fields are filled with 'Unknown'/NaN.
    stage_value_map overrides guesses for raw stage values (e.g. {'Onsite Loop': 'interviewed'})."""
    out = pd.DataFrame(index=raw.index)
    for fld in schema.ALL_COLUMNS:
        col = mapping.get(fld)
        out[fld] = raw[col] if col else pd.NA

    if mapping.get("applicant_id") is None:
        out["applicant_id"] = [f"row{i + 1}" for i in range(len(raw))]
    if mapping.get("req_id") is None:
        out["req_id"] = "ALL"
    if mapping.get("job_title") is None:
        out["job_title"] = "(not provided)"

    if mapping.get("stage_reached"):
        overrides = {str(k): v for k, v in (stage_value_map or {}).items()}
        out["stage_reached"] = raw[mapping["stage_reached"]].map(
            lambda v: overrides.get(str(v), guess_stage(v)) if pd.notna(v) else None
        )
    elif stage_flags and any(stage_flags.values()):
        out["stage_reached"] = stage_from_flags(raw, stage_flags)

    out["gender"] = normalize_category(out["gender"], _GENDER)
    out["race_ethnicity"] = normalize_category(out["race_ethnicity"], _RACE)
    out["age_band"] = normalize_category(out["age_band"], _AGE)
    for fld in ("ses_indicator", "source", "education_level"):
        out[fld] = normalize_category(out[fld], {})
    out["years_experience"] = pd.to_numeric(out["years_experience"], errors="coerce")
    out["application_date"] = pd.to_datetime(out["application_date"], errors="coerce")
    out["req_id"] = out["req_id"].astype(str)
    out["applicant_id"] = out["applicant_id"].astype(str)
    return out


def unmapped_stage_values(raw: pd.DataFrame, stage_col: str | None) -> list[str]:
    """Raw stage values FairHire can't place on the funnel by itself (user must map them)."""
    if not stage_col:
        return []
    vals = raw[stage_col].dropna().astype(str).unique()
    return sorted(v for v in vals if guess_stage(v) is None)


def validate(
    df: pd.DataFrame,
    raw_columns: list[str],
    mapping: dict[str, str | None],
    stage_flags: dict[str, str | None] | None = None,
) -> Report:
    r = Report()
    mapped_cols = {c for c in mapping.values() if c} | {c for c in (stage_flags or {}).values() if c}

    ids = identifier_columns(raw_columns, mapped_cols)
    if ids:
        r.errors.append(
            "Your file contains columns that look like personal identifiers: "
            + ", ".join(f"'{c}'" for c in ids)
            + ". Delete them from the file and upload again. FairHire never needs names, emails, "
            "phone numbers, addresses, or dates of birth."
        )

    has_stage = bool(mapping.get("stage_reached")) or bool(stage_flags and any(stage_flags.values()))
    if not has_stage:
        r.errors.append("Map a stage column (or yes/no columns per stage) so FairHire knows how far each "
                        "applicant got.")
    elif df["stage_reached"].isna().any():
        n = int(df["stage_reached"].isna().sum())
        r.errors.append(f"{n} rows have a stage value FairHire couldn't place. Map those values below.")

    for fld in ("gender", "race_ethnicity"):
        if not mapping.get(fld):
            r.warnings.append(f"No {fld.replace('_', '/')} column mapped: that part of the audit is skipped.")
    if not (mapping.get("gender") or mapping.get("race_ethnicity")
            or mapping.get("age_band") or mapping.get("ses_indicator")):
        r.errors.append("Map at least one demographic column (gender, race/ethnicity, age band, or SES).")

    if has_stage and df["stage_reached"].notna().all():
        order = {s: i for i, s in enumerate(schema.STAGES)}
        reached = df["stage_reached"].map(order)
        for i, st in enumerate(schema.STAGES[1:], start=1):
            if not (reached >= i).any():
                r.warnings.append(f"No applicant reached '{st}': that stage and later ones show no data.")
                break

    if len(df) < 100:
        r.warnings.append(f"Only {len(df)} rows. Most groups will be too small for a verdict (n < 30).")

    if mapping.get("applicant_id"):
        dup = df.duplicated(["applicant_id", "req_id"]).sum()
        if dup:
            r.warnings.append(f"{dup} duplicate applicant + requisition rows. Each applicant should appear "
                              "once per requisition; duplicates are counted twice.")

    for fld, allowed in (("gender", schema.GENDER_VALUES), ("race_ethnicity", schema.RACE_ETHNICITY_VALUES)):
        if mapping.get(fld):
            odd = sorted(set(df[fld].dropna()) - set(allowed))
            if odd:
                r.warnings.append(f"Unrecognized {fld.replace('_', '/')} values kept as their own groups: "
                                  + ", ".join(odd[:8]) + (" ..." if len(odd) > 8 else ""))
            unk = (df[fld].isin(schema.NON_RESPONSE_VALUES)).mean()
            if unk > 0.2:
                r.warnings.append(f"{unk:.0%} of {fld.replace('_', '/')} values are unknown or declined. "
                                  "If non-response differs by group, rates can be biased.")

    r.info.append(f"{len(df):,} applicant rows across {df['req_id'].nunique()} requisition(s).")
    return r
