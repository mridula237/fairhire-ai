# Legal notes (checked Oct 29, 2026)

Summary of the laws FairHire AI is designed around. **Not legal advice.** Re-check before quoting in a report.

## Federal: Uniform Guidelines, 29 CFR 1607.4(D)

- **Four-fifths rule.** A group's selection rate below 80% of the highest group's rate is *generally* evidence of adverse impact.
- **Smaller gaps can still count** when they are significant both statistically and practically, or when the employer discouraged applicants.
- **Larger gaps may not count** when they rest on small numbers and are not statistically significant, or when special recruiting made the applicant pool unusual.
- **Too-small samples:** agencies may look at a longer time period or at the same procedure used elsewhere.

**FairHire implication:** report AIR *with* significance tests and confidence intervals, apply a min-n guard, and let the user choose the reference group.

Source: [29 CFR 1607.4 (Cornell LII)](https://www.law.cornell.edu/cfr/text/29/1607.4)

## New York City: Local Law 144 (enforced since July 5, 2023)

- **Scope:** employers using an automated employment decision tool (AEDT) for hiring or promotion in NYC.
- **What the audit must report:**
  - selection rate (or scoring rate)
  - impact ratio versus the highest category
  - number of applicants per category
- **Categories:** sex, race/ethnicity, and intersectional combinations of the two.
- **Small categories:** a category under 2% of the data may be left out of the impact ratio calculation, but its count and rate must still be reported.
- **Unknowns:** unknown sex or race is reported as a separate count and kept out of the ratios.
- **Independent auditor:** the audit must be done by an auditor independent of the employer and the tool vendor. That is why ChiEAC frames its NYC offer as "audit readiness".
- **Obligations on the employer:** publish a summary of the audit results and notify candidates that the tool is used.
- **Penalties:** $500 for a first violation, up to $1,500 for each one after.

**FairHire implication:**
- intersectional sex × race view
- 2% exclusion flag
- unknowns reported separately
- output table: category · n · selected · rate · impact ratio

Sources: [DCWP final rule (Sidley PDF)](https://www.sidley.com/-/media/resource-pages/ai-monitor/laws-and-regulations/20230406-nyc-dcwp-notice-of-adoption-of-final-rule-for-local-law-144.pdf?la=en), [Paul Hastings summary](https://www.paulhastings.com/insights/client-alerts/the-department-of-consumer-and-worker-protection-issues-final-rules)

## Illinois: Public Act 103-0804 (HB 3773), in force since Jan 1, 2026

- **Discriminatory effect:** it is a civil rights violation to use AI in employment decisions (recruitment, hiring, promotion, discipline, and so on) in a way that discriminates on a protected class, **even unintentionally**.
- **Zip codes:** it is a violation to **use zip codes as a proxy for protected classes**.
- **Notice:** employers must tell applicants and employees when AI is used.
- **Rules status:**
  - IDHR published proposed notice and recordkeeping rules on May 15, 2026.
  - It withdrew and postponed them on June 2, 2026.
  - As of June 2026 no final rules exist, but the law itself still applies.

**FairHire implication:**
- Module 3's proxy scan should flag any ZIP-derived feature explicitly.
- The Illinois AI-in-Hiring Readiness Check stands on the statute alone. Re-check IDHR before selling it.

Sources: [Ogletree (Jun 4, 2026)](https://ogletree.com/insights-resources/blog-posts/illinois-postpones-proposed-regulations-on-ai-in-employment/), [Gordon Rees](https://www.grsm.com/insight/amendments-to-illinois-human-rights-act-to-regulate-use-of-ai-in-employment-decisions/), [Baker Hostetler](https://www.bakerlaw.com/insights/illinois-proposes-specific-notice-requirements-for-employers-who-use-ai-in-hiring-and-employment-decisions/)

## Colorado: SB 24-205 replaced by SB 26-189

- **The brief is out of date here.** SB 24-205 never took effect in its original form.
  - Enforcement was stayed on April 27, 2026.
  - **SB 26-189** (signed May 14, 2026) repealed and re-enacted it as a narrower "automated decision-making technology" law.
- **Effective date:** January 1, 2027. Enforcement depends on Attorney General rulemaking, due by Jan 1, 2027.
- **Employment:** hiring is a "consequential decision". Deployers must:
  - give notice before using ADMT
  - explain adverse outcomes in plain language within 30 days
  - offer correction and human-review rights
  - keep records for 3 years
- **What was dropped:** SB 24-205's standalone annual impact assessments and broad algorithmic-discrimination duty.
- **Enforcement:** by the AG only, up to $20,000 per violation.

**FairHire implication:** do not market the tool as a "Colorado impact assessment". Position it as supporting documentation and explanation duties. Re-check the AG's ADMT page before Week 7 pricing.

Sources: [AI Compliance Atlas](https://aicomplianceatlas.com/law/colorado-ai-act), [Greenberg Traurig (Sep 2025)](https://www.gtlaw.com/en/insights/2025/9/colorado-delays-comprehensive-ai-law-with-further-changes-anticipated)

## Flag for Benjamin

The brief's Colorado section and the Section 7 "Illinois Readiness Check" assume older facts. Raise this in the Week 1 email.
