# FairHire AI

**Open-source hiring funnel disparity audit.** Upload an anonymized applicant file (or use demo data) and see where a hiring process treats groups differently: at which stage, for which groups, how large the gap is, whether it is statistically meaningful, and what to change.

Built with the [Chicago Education Advocacy Cooperative (ChiEAC)](https://chieac.org). Lead developer: Mridula Kalaiselvan.

> FairHire AI provides statistical analysis and educational information. It is not legal advice and does not replace an independent bias audit where a law requires one.

**Live app:** _coming in week 1_ · **Status:** v0 (week 1 of 8)

## Modules (roadmap)

| # | Module | Target |
|---|---|---|
| 1 | Hiring Funnel Disparity Audit: selection rates, adverse impact ratios, four-fifths rule, z-test / Fisher / chi-square / SD test, Wilson CIs, intersectional view | v0.1 (Nov 13) |
| 2 | Job Description Language Audit: gender-coded, age-coded, exclusionary language, reading level, rewrites | v0.2 (Nov 20) |
| 3 | Model Fairness & Explainability Check: LightGBM, Fairlearn MetricFrame, SHAP, proxy scan (diagnostic only) | v0.3 (Nov 27) |
| 4 | Equity Assessment Report: HTML/PDF with risk ratings, recommendations, 90-day plan | v0.4 (Dec 4) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                                   # tests
PYTHONPATH=src python -m fairhire.make_demo_data   # writes data/demo_hiring_data.csv
streamlit run app.py
```

Inject your own disparities:

```bash
PYTHONPATH=src python -m fairhire.make_demo_data --n 20000 \
  --disparity "race_ethnicity|Black or African American|screened|0.75" \
  --disparity "gender|Female|offered|0.70"
```

## Data

- Template: [`data/data_template.csv`](data/data_template.csv)
- Dictionary: [`docs/data_dictionary.md`](docs/data_dictionary.md)

## Guardrails

- Anonymized data only; identifier-like columns are rejected.
- Nothing is stored. Uploads live in session memory.
- Protected attributes measure outcomes; they are never model features.
- No inference of race, gender, or ethnicity. Ever.
- Minimum group sizes and uncertainty next to every estimate.
- No legal determinations.

## License

MIT. See [LICENSE](LICENSE).
