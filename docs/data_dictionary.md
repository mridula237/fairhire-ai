# FairHire AI Data Dictionary (v0)

One row per **applicant per requisition**. CSV or XLSX. Template: [`data/data_template.csv`](../data/data_template.csv).

> **Strip identifiers first.** Files with columns that look like name, email, phone, address, SSN, or date of birth are rejected. Nothing you upload is stored.

| Column | Type | Required | Allowed values / format | Notes |
|---|---|---|---|---|
| `applicant_id` | string | Yes | Any anonymized ID (e.g. `A000123`) | Created by the uploader. Never a name, email, or employee number. |
| `req_id` | string | Yes | e.g. `REQ-101` | One requisition per posting. Audits run per requisition and pooled. |
| `job_title` | string | Yes | Free text | |
| `stage_reached` | ordered category | Yes | `applied` < `screened` < `interviewed` < `offered` < `hired` | The furthest stage this applicant reached. Boolean per-stage columns (`screened=1/0`, ...) will be accepted as an alternative via the column mapper (week 3). |
| `gender` | category | Yes (for audit) | `Female`, `Male`, `Nonbinary`, `Prefer not to say`, `Unknown` | As self-reported on the voluntary EEO form. Non-response is reported separately, never used as the reference group. |
| `race_ethnicity` | category | Yes (for audit) | EEO-1: `Hispanic or Latino`, `White`, `Black or African American`, `Asian`, `Native Hawaiian or Other Pacific Islander`, `American Indian or Alaska Native`, `Two or More Races`, `Unknown` | **Never inferred by the app.** |
| `age_band` | category | Optional | `Under 40`, `40 and over`, `Unknown` | ADEA protects 40 and over. |
| `ses_indicator` | category | Optional | e.g. `First-generation college`, `Not first-generation`, `Unknown` | Any socioeconomic proxy the uploader already holds (first-gen, Pell, ZIP income tier, public vs private high school). |
| `source` | category | Optional | `Referral`, `Job board`, `Campus`, `Agency`, `Company website` | Referral-heavy pipelines are a classic driver of disparity. |
| `education_level` | ordinal | Optional | `High school` < `Associate` < `Bachelor` < `Master` < `Doctorate` | Module 3 feature; also tested as a proxy. |
| `years_experience` | numeric | Optional | ≥ 0 | Module 3 feature; also tested as a proxy. |
| `application_date` | date | Optional | `YYYY-MM-DD` | Enables trend by quarter. |

## How the audit uses these columns

- **Audit attributes** (`gender`, `race_ethnicity`, `age_band`, `ses_indicator`) are used only to **measure** outcomes. They are never model features.
- **Selection rate** at a stage = applicants who reached that stage ÷ applicants who reached the previous stage.
- **Minimum group size:** groups with fewer than 30 applicants at a stage are shown with a caution label and no four-fifths verdict.

## Synthetic demo data

`python -m fairhire.make_demo_data` writes `data/demo_hiring_data.csv` (5,000 rows) plus `demo_hiring_data.ground_truth.json`, which records every injected disparity. Default: Hispanic or Latino applicants pass screened → interviewed at 0.65× everyone else's rate. All other group × stage multipliers are 1.0.
