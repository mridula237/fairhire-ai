"""Fairlearn quick start on FairHire demo data.

Run:  PYTHONPATH=src python notebooks/fairlearn_quickstart.py

Part 1 cross-checks our AIR against Fairlearn on the ACTUAL hiring decisions.
Part 2 is a Module 3 preview: train a model on past decisions and ask whether
what it learned is fair.

Fairlearn in 4 ideas:
  1. MetricFrame(metrics, y_true, y_pred, sensitive_features) = any metric, split by group.
     .by_group = per-group table; .difference() / .ratio() = gap between best and worst group.
  2. selection_rate = share predicted/selected as 1. Its ratio across groups = an AIR
     (vs. the highest group, i.e. the four-fifths rule's definition).
  3. demographic_parity_ratio/difference = selection-rate gap (ignores whether the
     decision was "right"); equalized_odds_* = gap in TPR/FPR (needs real outcomes).
  4. Mitigation (ExponentiatedGradient, ThresholdOptimizer) retrains or re-thresholds a
     model to shrink a chosen gap, usually at some accuracy cost. Module 3, Week 5.
"""

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    false_negative_rate,
    selection_rate,
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from fairhire import schema
from fairhire.fairhire_stats import audit_stage
from fairhire.make_demo_data import GeneratorConfig, generate

df = generate(GeneratorConfig(n=20000, seed=42))
order = {s: i for i, s in enumerate(schema.STAGES)}
reached = df["stage_reached"].map(order)

# --- Part 1: cross-check AIR on actual decisions (screened -> interviewed) ----------------
screened = df[reached >= order["screened"]]
known = ~screened["race_ethnicity"].isin(schema.NON_RESPONSE_VALUES)
screened = screened[known]
y = (screened["stage_reached"].map(order) >= order["interviewed"]).astype(int)

mf = MetricFrame(metrics=selection_rate, y_true=y, y_pred=y, sensitive_features=screened["race_ethnicity"])
print("Part 1: interview rate among screened applicants, by group (Fairlearn)")
print(mf.by_group.round(3).to_string())
print(f"Fairlearn ratio (worst / best group): {mf.ratio():.3f}")

ours = audit_stage(df, "race_ethnicity", "interviewed", reference="highest", min_n=0)
print(f"FairHire min AIR (min_n=0 to match):  {ours['air'].min():.3f}")
print("These match: same definition. FairHire adds min-n, pooled reference, and notes.\n")

# --- Part 2: what does a model learn from past hiring decisions? ------------------------
features = ["education_level", "years_experience", "source"]  # non-protected only
X = df[features].copy()
X["education_level"] = X["education_level"].map({e: i for i, e in enumerate(schema.EDUCATION_VALUES)})
X = X.join(pd.get_dummies(df["source"], prefix="src", dtype=int)).drop(columns="source")
y_hired = (reached == order["hired"]).astype(int)
group = df["race_ethnicity"]

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    X, y_hired, group, test_size=0.3, random_state=0, stratify=y_hired
)
model = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
y_pred = (proba >= np.quantile(proba, 1 - y_tr.mean())).astype(int)  # select same share as history

mf2 = MetricFrame(
    metrics={"selection_rate": selection_rate, "false_negative_rate": false_negative_rate},
    y_true=y_te, y_pred=y_pred, sensitive_features=g_te,
)
print("Part 2: model trained on non-protected features only")
print(mf2.by_group.round(3).to_string())
dpr = demographic_parity_ratio(y_te, y_pred, sensitive_features=g_te)
print(f"Demographic parity ratio of the MODEL: {dpr:.3f}")
print(
    "\nRead this honestly:\n"
    " * The model is weak (false negative rate ~0.9 for everyone): only ~5% are hired and education,\n"
    "   experience and source barely predict it in the demo data. A weak model can't tell you much.\n"
    " * Selection rates are similar across large groups because the features are distributed the\n"
    "   same for every group (by design, no proxies yet).\n"
    " * The parity ratio is dragged down by tiny groups (e.g. ~20 people). Same lesson as the\n"
    "   four-fifths rule: never read a fairness ratio without group sizes.\n"
    "Module 3 TODO: min-group guard (>=300 rows, >=30 hires per group), a proxy feature in the\n"
    "generator (e.g. referral share differing by group) so there is something real to detect."
)
