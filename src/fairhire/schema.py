"""Canonical FairHire AI schema: column names, stage order, allowed categories.

Everything else in the project (template, generator, validator, stats engine)
imports from here so the schema lives in exactly one place.
"""

from itertools import pairwise

STAGES = ["applied", "screened", "interviewed", "offered", "hired"]

# Stage transitions audited in Module 1: (from_stage, to_stage)
TRANSITIONS = list(pairwise(STAGES))

REQUIRED_COLUMNS = [
    "applicant_id",
    "req_id",
    "job_title",
    "stage_reached",
    "gender",
    "race_ethnicity",
]

OPTIONAL_COLUMNS = [
    "age_band",
    "ses_indicator",
    "source",
    "education_level",
    "years_experience",
    "application_date",
]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

# Protected / audit attributes: used to MEASURE outcomes, never as model features.
AUDIT_ATTRIBUTES = ["gender", "race_ethnicity", "age_band", "ses_indicator"]

GENDER_VALUES = ["Female", "Male", "Nonbinary", "Prefer not to say", "Unknown"]

# EEO-1 race/ethnicity categories plus "Unknown". Never inferred by the app.
RACE_ETHNICITY_VALUES = [
    "Hispanic or Latino",
    "White",
    "Black or African American",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "American Indian or Alaska Native",
    "Two or More Races",
    "Unknown",
]

AGE_BAND_VALUES = ["Under 40", "40 and over", "Unknown"]  # ADEA line

SES_VALUES = ["First-generation college", "Not first-generation", "Unknown"]

SOURCE_VALUES = ["Referral", "Job board", "Campus", "Agency", "Company website"]

EDUCATION_VALUES = ["High school", "Associate", "Bachelor", "Master", "Doctorate"]

# Values reported separately and never used as a reference group.
NON_RESPONSE_VALUES = {"Unknown", "Prefer not to say"}

# Column-name fragments that suggest personal identifiers. The week-3 validator
# rejects any upload with a column matching one of these.
IDENTIFIER_PATTERNS = [
    "name", "email", "e-mail", "phone", "mobile", "address", "street",
    "ssn", "social_security", "dob", "date_of_birth", "birth",
]

MIN_GROUP_N = 30  # small-sample guard: no four-fifths verdict below this
