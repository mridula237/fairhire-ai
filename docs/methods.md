# FairHire AI: Statistical Methods (v0.1)

This document explains how FairHire AI measures hiring disparities, so a reviewer can check every number by hand. The code is in `src/fairhire/fairhire_stats.py`, and hand-calculated tests are in `tests/test_significance.py`.

> FairHire AI provides statistical analysis, not legal advice. A flag means "look closer", not "discrimination occurred".

## 1. Unit of analysis: stage transitions

Each applicant has a `stage_reached`: applied < screened < interviewed < offered < hired. For each transition, for example screened → interviewed:

- **At risk (n):** applicants who reached the previous stage.
- **Selected:** the at-risk applicants who reached this stage.
- **Selection rate:** selected ÷ n.

Rates are computed by stage, not overall, because an overall hire rate hides *where* a gap occurs.

## 2. Adverse impact ratio (AIR) and the four-fifths rule

AIR = group's selection rate ÷ reference rate. Under the Uniform Guidelines, 29 CFR 1607.4(D), a rate below four-fifths (0.80) of the highest group's rate is *generally* regarded as evidence of adverse impact. NYC Local Law 144 reports the same ratio as its "impact ratio".

**Reference options** (`reference=`):

| Option | Compares each group to | When to use |
|---|---|---|
| `highest` (default) | The highest-rate *eligible* group | Matches the four-fifths rule and LL144 |
| `pooled` | All other known-response applicants combined | More stable, and recommended alongside `highest` |
| a group name | That group, e.g. `White` | When a specific comparison is required |

**Why the pooled option exists.** The highest group is a maximum, so a small group that gets lucky can become the benchmark and drag every other ratio down. In the 5,000-row demo, the injected AIR of 0.65 measured 0.52 against the highest group, a 71-person group whose rate was high by chance, but 0.61 against everyone else pooled. FairHire limits this in two ways: only groups with n ≥ 30 can be the reference, and the pooled option is offered alongside.

## 3. Significance tests

1607.4(D) says smaller differences may still be adverse impact when they are "significant in both statistical and practical terms". It also says larger differences may not be, when they rest on small numbers and are not statistically significant. FairHire therefore reports both.

| Test | What it answers | Details |
|---|---|---|
| **Two-proportion z-test** | Is this group's rate different from the reference? | Pooled SE: √(p̂(1−p̂)(1/n₁ + 1/n₂)), two-sided |
| **Fisher's exact test** | Same question, for small samples | Used instead of z when any expected 2×2 cell count < 5 |
| **Chi-square test of independence** | Do rates differ across *any* groups at this stage? | All eligible groups × (selected, not selected), no continuity correction; one p-value per stage |
| **Standard deviation (Hazelwood) test** | How many SDs below expected is this group? | (observed − n·p) ÷ √(n·p·(1−p)), where p = overall rate. Courts have treated gaps of more than 2–3 SD as suspect (*Hazelwood School District v. United States*, 1977) |

### Multiple comparisons

An audit runs many tests at once: groups × stages, and up to about 50 for intersectional views. At α = 0.05, some tests will come out "significant" by chance alone. Without correction, the 5,000-row demo flagged Male | Hispanic or Latino and Male | Black or African American at the **offer** stage, where no gap was injected.

FairHire therefore applies the **Holm–Bonferroni** correction. `audit_stage` adjusts within one stage, and `audit_funnel` and the CLI adjust across the whole funnel. Verdicts use `p_adjusted`, and the raw `p_value` is kept for transparency.

Checked across 20 random demo datasets: the injected gap was found 20/20 times, and 2 of 40 runs (5%) had any false alarm, which is the rate the correction is designed to allow.

## 4. Uncertainty

- **Wilson 95% interval** for each group's rate. Unlike the simple ±1.96·SE interval, it stays inside [0, 1] and behaves well for small n and rates near 0 or 1.
- **Katz log-method 95% interval** for the AIR: exp(log AIR ± 1.96·√(1/x₁ − 1/n₁ + 1/x₂ − 1/n₂)). If the interval crosses 0.80, the four-fifths verdict is fragile.

## 5. Verdicts

| Verdict | Condition |
|---|---|
| **adverse impact indicated** | AIR < 0.80 **and** adjusted p < 0.05 |
| **inconclusive: not significant** | AIR < 0.80 but adjusted p ≥ 0.05, often because of small numbers |
| **significant gap above 4/5** | AIR ≥ 0.80, but adjusted p < 0.05 and the group's rate is below the reference |
| **no adverse impact indicated** | Otherwise |
| **no verdict** | Non-response group, n < 30, or the group *is* the reference |

## 6. Guardrails

- **Minimum group size:** groups with fewer than 30 people at a stage get a caution note and no verdict. They can never be the reference.
- **Non-response** ("Unknown", "Prefer not to say") is reported separately and never used in ratios or as a reference, as LL144 requires.
- **LL144 2% rule:** categories under 2% of the data are noted, because LL144 lets an auditor exclude them from impact ratios.
- **Intersectional view:** sex × race/ethnicity (`--intersectional`), as LL144 requires.
- **No inference:** demographic fields come only from what applicants self-reported. Nothing is guessed.

## 7. Validation

- **Hand-calculated tests** for every building block, using the worked example A = 50/100 vs B = 30/100:
  - z = −2.887, p = 0.0039
  - χ² = 8.33
  - SD = −2.04
  - Wilson CI for 30/100 = (0.219, 0.396)
  - Katz CI for the AIR = (0.42, 0.86)
  - Fisher p = 0.0059, cross-checked by summing hypergeometric terms
- **Ground-truth tests:** synthetic data with known injected gaps. The engine recovers the injected AIR, and groups with no injected gap stay near 1.0.
- **Cross-check:** Fairlearn's `MetricFrame` selection-rate ratio equals FairHire's AIR (0.604 on the 20k demo, `notebooks/fairlearn_quickstart.py`).

## 8. Known limitations

- **Selection only.** A gap can have a lawful, job-related explanation. The four-fifths rule is a screening rule, not proof of discrimination.
- **Self-reported fields have missing data.** If non-response differs by group, the rates can be biased.
- **Statistical significance grows with n.** On large data, a tiny gap can be "significant". That is why the verdict also requires AIR < 0.80, and why "significant gap above 4/5" is its own category, not a finding of adverse impact.
- **No control for qualifications.** Comparing like with like is part of Module 3 (model fairness).

## References

- Uniform Guidelines on Employee Selection Procedures, [29 CFR 1607.4(D)](https://www.law.cornell.edu/cfr/text/29/1607.4)
- NYC Local Law 144 and DCWP final rule: see `docs/legal_notes.md`
- *Hazelwood School District v. United States*, 433 U.S. 299 (1977)
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *JASA* 22(158).
- Katz, D., et al. (1978). Obtaining confidence intervals for the risk ratio in cohort studies. *Biometrics* 34(3).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2).
