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

---

# Module 2: Job Description Language Audit (v0.2)

Code: `src/fairhire/jd_audit.py`. Flag dictionary: `src/fairhire/jd_flags.yaml`. Tests: `tests/test_jd_audit.py`. Batch scoring: `python -m fairhire.jd_batch <folder>`.

Everything is rule-based and explainable. Each highlight and each point lost traces back to a specific phrase.

## Gender-coded wording

FairHire uses the masculine- and feminine-coded word stems from Kat Matfield's Gender Decoder, which implement **Gaucher, Friesen & Kay (2011)**. That study found that masculine-coded job ads made jobs less appealing to women and lowered their sense of belonging. It did not find that the wording changed women's view of their own skills. A stem matches any word that begins with it; for example, *compet* matches competitive and competition. The stem lists are used as published, so results match the Gender Decoder. The only exceptions are a few unrelated words that happen to share a stem: kindergarten, committee, and Connecticut. Three more forms are excluded because postings use them in boilerplate with a different meaning: *dependents* ("coverage for your dependents"), *depending* ("depending on experience"), and *individuals* ("individuals with disabilities"). This was found while testing 10 real Chicago postings in Nov 2026.

Rating:

| Difference (masculine − feminine) | Rating |
|---|---|
| 3 or more | strongly masculine-coded |
| 1 to 2 | masculine-coded |
| 0 | neutral |
| −1 to −2 | feminine-coded |
| −3 or less | strongly feminine-coded |

## Flag dictionary

The YAML file defines 18 flags in 6 categories:

| Category | Examples |
|---|---|
| Age-coded | "digital native", "young team", "recent grads only", experience caps |
| Exclusionary | "culture fit", "rockstar", "native English speaker", "U.S. citizens only", gendered pronouns and titles, grooming rules |
| Disability | Lifting or standing requirements without an accommodation statement, driver's license requirements |
| Credential | "Degree required" with no "or equivalent experience" |
| Legal | Criminal-history screens in the posting |

Each flag carries a severity, a plain-language reason, and a suggested fix. Severity:
- **High:** likely to screen people out on a protected trait, or raises legal risk.
- **Medium:** known to discourage some groups.
- **Low:** a style or clarity issue.

## Pay transparency

Since Jan 1, 2025, Illinois employers with 15 or more employees must include the pay scale and a general description of benefits in job postings (Illinois Equal Pay Act, 820 ILCS 112/10(b-25)). FairHire detects a pay range with patterns such as "$X–$Y" and "$X per hour", and detects benefits with keywords.

## Readability

FairHire computes the **Flesch–Kincaid grade level** and the **Flesch reading ease** score with a built-in heuristic syllable counter, so nothing has to be downloaded at runtime. Bullet lines count as sentences. FairHire suggests grade 10 or below.

## JD Equity Score (0–100)

The score starts at 100, and every deduction is itemized in the app and in the downloadable report.

| Issue | Deduction | Cap |
|---|---|---|
| Each flagged phrase | high −12, medium −6, low −3 | −45 total |
| Masculine-coded wording outweighs feminine-coded by 3 or more | −3 per word of imbalance | −15 |
| Reading grade above 10 | −2 per grade | −15 |
| No pay range | −10 | |
| Pay range listed but no benefits | −3 | |
| Over 900 words | −5 | |

The weights are a transparent design choice, not a validated psychometric scale. The score helps compare drafts and postings. It does not measure legal compliance.

## Suggested rewrite

The rewrite applies only the word-level swaps marked `replace:` in the YAML, such as "rockstar" → "skilled" and "culture fit" → "culture add". It also adds "or equivalent experience" to a degree requirement. Flags that need human judgment, such as pronouns, physical requirements, and citizenship, are left in place and listed for review.

## Limitations

- Word lists can't read context. "Lead" in "lead paint abatement" still counts as masculine-coded.
- The research on gendered wording is about **appeal to applicants**. It doesn't show that any single word is discriminatory.
- Flags point to risk; they are not legal conclusions. Some requirements, such as lifting or citizenship, may be legitimate and legally required for a specific job.

## References

- Gaucher, D., Friesen, J., & Kay, A. C. (2011). Evidence that gendered wording in job advertisements exists and sustains gender inequality. *Journal of Personality and Social Psychology*, 101(1), 109–128.
- Matfield, K. Gender Decoder (MIT License). https://github.com/lovedaybrooke/gender-decoder
- Kincaid, J. P., et al. (1975). Derivation of new readability formulas for Navy enlisted personnel. Naval Technical Training Command.
- Illinois Department of Labor, Pay Transparency: https://labor.illinois.gov/pay

---

# Module 3: Model Fairness & Explainability Check (v0.3)

Code: `src/fairhire/model_fairness.py`. Tests: `tests/test_model_fairness.py`. Benchmark: `scripts/benchmark_adult.py`.

> **Diagnostic only.** Module 3 trains a model on past hiring decisions to show what those decisions rewarded. It is never a screening tool, and the app says so on every screen.

## Question

"If we trained a model on our past decisions, what would it learn, and would it treat groups differently?" More and more employers use automated screening tools, and those tools are trained on exactly this kind of history.

## Method

1. **Target.** The target is whether an applicant reached the chosen stage. The default is *screened*, where most automated screening happens.
2. **Inputs.** Only non-protected columns are used as inputs: source, education level, years of experience, and optionally job title or requisition. Gender, race/ethnicity, age band and SES are **refused as inputs** and used only to measure outcomes.
3. **Model.** LightGBM with 5-fold stratified cross-validation. Every applicant is scored by a model that never saw them (out-of-fold predictions). The threshold is set so the model selects the same share of applicants as history did.
4. **Group metrics.** Fairlearn `MetricFrame`: selection rate, true-positive, false-positive and false-negative rate by group; demographic parity ratio and difference; equalized odds difference. Each group's model rate is shown next to its **past rate**, so you can see whether the model widens or narrows historical gaps. Plain-language comparisons use groups with 100+ applicants, because smaller groups are too noisy to summarize.
5. **Explainability.** SHAP values come from LightGBM's built-in TreeSHAP (`pred_contrib=True`). This gives exact Shapley values for tree models with no extra package. The app shows mean |SHAP| per input (global importance) and mean SHAP per group for the top input (which groups it pushed up or down).
6. **Proxy scan.** The scan checks how strongly each input is associated with each protected attribute:
   - Categorical inputs: bias-corrected Cramér's V (Bergsma 2013).
   - Numeric inputs: correlation ratio η.
   - Non-response is excluded.
   - Levels: ≥ 0.30 strong, ≥ 0.15 moderate, ≥ 0.05 weak.
7. **Mitigation what-if.** Fairlearn's `ThresholdOptimizer` (demographic parity) is run cross-fitted, and the app reports the group ratio and accuracy before and after. This is shown as a what-if, not a recommendation to deploy. The better fix is to change the practice the model learned from.

## Guards

The check won't run unless:
- there are at least **300 rows**, and
- at least **2 known-response groups** each have **30 or more** applicants who reached the target stage.

If the model's AUC is below 0.60, the app warns that the inputs explain little and that gaps should be read with extra caution.

## Validation

- **Benchmark against published results.** `scripts/benchmark_adult.py` reproduces the Fairlearn v0.9 quickstart on UCI Adult (48,842 rows). All 9 published numbers match within ±0.001, except mitigated Male (0.1708 vs 0.1715):
  - accuracy: overall 0.8444, Female 0.9251, Male 0.8043
  - selection rate: overall 0.1639, Female 0.0635, Male 0.2136
  - ExponentiatedGradient selection rate: Female 0.1553, Male 0.1708
- **Planted-proxy test.** The demo generator's `referral_proxy=True` gives White applicants referrals 5× as often and has past screeners favor referrals (×1.8 pass rate). Race is never a model input.

  Module 3 finds:
  - *source* is the top input, and a moderate proxy for race (V ≈ 0.17).
  - SHAP for *source* is positive for White applicants and negative for every other group.
  - The model **widens** the historical gap: among groups with 100+ applicants, the lowest-to-highest ratio goes from 0.79 to 0.72.
  - ThresholdOptimizer moves the group ratio to about 0.82 with no accuracy loss.

  **Control:** with no proxy, no input is flagged and the ratio is unchanged (0.91 → 0.91).

## Limitations

- Out-of-fold models approximate what a vendor model trained on this history would learn. They are not the vendor's actual model.
- Association isn't causation. A strong proxy is a reason to examine a practice, not proof of discrimination.
- With weak inputs (low AUC), group differences in model behavior are mostly noise.

## References

- Bird, S., et al. (2020). Fairlearn: A toolkit for assessing and improving fairness in AI. Microsoft Research Technical Report MSR-TR-2020-32.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
- Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. *Journal of the Korean Statistical Society*, 42(3), 323–328.
- Hardt, M., Price, E., & Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
- Fairlearn quickstart (v0.9): https://fairlearn.org/v0.9/quickstart.html
