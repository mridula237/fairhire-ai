"""Module 3: Model Fairness & Explainability Check (DIAGNOSTIC ONLY, not a hiring tool).

Question it answers: "If we trained a model on our past hiring decisions, what would it learn,
and would it treat groups differently?" It is a mirror held up to historical decisions, never a
screening tool. The app labels it that way and refuses to run on data too small to say anything.

Pipeline
1. Target = applicant reached `target_stage` (default: screened, where most automated screening
   happens). Features = NON-protected columns only (source, education, experience, ...).
   Protected attributes are used only to MEASURE, never as inputs.
2. LightGBM, 5-fold stratified cross-validation -> out-of-fold predictions (every applicant is
   scored by a model that never saw them). Threshold = the historical selection rate.
3. Fairlearn MetricFrame by group: selection rate, true/false positive rate, false negative rate;
   demographic parity ratio/difference, equalized odds difference.
4. Explainability: SHAP values from LightGBM's built-in TreeSHAP (pred_contrib=True, no extra
   package): global mean |SHAP| per feature and mean SHAP per feature by group.
5. Proxy scan: association of every feature with every protected attribute
   (Cramér's V for categorical, correlation ratio eta for numeric).
6. Optional mitigation comparison: Fairlearn ThresholdOptimizer (demographic parity).

Guards: >= 300 rows, and >= 2 known-response groups with >= 30 positive outcomes each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from scipy import stats
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fairhire import schema

MIN_ROWS = 300
MIN_POSITIVES_PER_GROUP = 30
STABLE_GROUP_N = 100  # for plain-language comparisons; smaller groups are too noisy to summarize
DEFAULT_FEATURES = ["source", "education_level", "years_experience"]
CANDIDATE_FEATURES = ["source", "education_level", "years_experience", "job_title", "req_id"]
LABEL = "Diagnostic only: this model shows what past decisions rewarded. It must never be used to screen people."

LGB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "verbose": -1,
    "n_jobs": 1,  # small server: one core, predictable memory
}


class GuardError(ValueError):
    """Data too small or too thin to support a fairness diagnosis."""


@dataclass
class ModelReport:
    target_stage: str
    attribute: str
    features: list[str]
    n: int
    base_rate: float
    auc: float
    accuracy: float
    by_group: pd.DataFrame                       # selection_rate, tpr, fpr, fnr, n, actual_rate (history)
    dp_ratio: float
    dp_difference: float
    eo_difference: float
    shap_global: pd.Series                       # mean |SHAP| per feature, sorted
    shap_by_group: pd.DataFrame                  # mean SHAP per feature (cols) per group (rows)
    proxies: pd.DataFrame                        # feature x attribute association
    mitigation: dict | None = None
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Association measures (proxy scan)
# ---------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) between two categorical series."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False).statistic
    n = table.to_numpy().sum()
    r, k = table.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(math.sqrt(phi2 / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta: how much of a numeric feature's variance the groups explain."""
    df = pd.DataFrame({"c": categories, "v": pd.to_numeric(values, errors="coerce")}).dropna()
    if df.empty or df["v"].var() == 0:
        return 0.0
    grand = df["v"].mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in df.groupby("c")["v"])
    total = ((df["v"] - grand) ** 2).sum()
    return float(math.sqrt(between / total)) if total > 0 else 0.0


def proxy_scan(df: pd.DataFrame, features: list[str], attributes: list[str]) -> pd.DataFrame:
    """Rows = features, columns = protected attributes, values = association (0 to 1).
    Non-response rows are excluded so 'Unknown' can't create fake associations."""
    out = pd.DataFrame(index=features, columns=attributes, dtype=float)
    for a in attributes:
        known = ~df[a].isin(schema.NON_RESPONSE_VALUES)
        for f in features:
            col = df.loc[known, f]
            if pd.api.types.is_numeric_dtype(col):
                out.loc[f, a] = correlation_ratio(df.loc[known, a], col)
            else:
                out.loc[f, a] = cramers_v(col.astype(str), df.loc[known, a])
    return out


def proxy_level(v: float) -> str:
    return "strong" if v >= 0.3 else "moderate" if v >= 0.15 else "weak" if v >= 0.05 else "none"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _design_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f in features:
        col = df[f]
        if f == "education_level":
            X[f] = col.map({e: i for i, e in enumerate(schema.EDUCATION_VALUES)})
        elif pd.api.types.is_numeric_dtype(col):
            X[f] = pd.to_numeric(col, errors="coerce")
        else:
            X[f] = col.astype("category")
    return X


def check_guards(df: pd.DataFrame, y: pd.Series, attribute: str) -> list[str]:
    problems = []
    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows; the model check needs at least {MIN_ROWS}.")
    known = ~df[attribute].isin(schema.NON_RESPONSE_VALUES)
    pos = y[known].groupby(df.loc[known, attribute]).sum()
    ok = pos[pos >= MIN_POSITIVES_PER_GROUP]
    if len(ok) < 2:
        problems.append(
            f"Needs at least 2 groups with {MIN_POSITIVES_PER_GROUP}+ applicants who reached the target stage; "
            f"found {len(ok)}."
        )
    return problems


def run_model_check(
    df: pd.DataFrame,
    attribute: str = "race_ethnicity",
    target_stage: str = "screened",
    features: list[str] | None = None,
    min_group_n: int = schema.MIN_GROUP_N,
    seed: int = 0,
    mitigate: bool = True,
) -> ModelReport:
    import lightgbm as lgb  # imported here so the rest of the app loads fast

    features = [f for f in (features or DEFAULT_FEATURES) if f in df.columns]
    banned = set(schema.AUDIT_ATTRIBUTES) & set(features)
    if banned:
        raise ValueError(f"Protected attributes can't be model features: {sorted(banned)}")
    if not features:
        raise GuardError("No usable non-protected features (e.g. source, education_level, years_experience).")

    order = {s: i for i, s in enumerate(schema.STAGES)}
    y = (df["stage_reached"].map(order) >= order[target_stage]).astype(int)
    problems = check_guards(df, y, attribute)
    if problems:
        raise GuardError(" ".join(problems))

    X = _design_matrix(df, features)
    oof = np.zeros(len(df))
    contrib = np.zeros((len(df), len(features)))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        contrib[te] = m.predict(X.iloc[te], pred_contrib=True)[:, : len(features)]  # last col = bias

    base_rate = float(y.mean())
    threshold = float(np.quantile(oof, 1 - base_rate))
    y_pred = (oof >= threshold).astype(int)

    group = df[attribute].astype(str)
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate,
                 "fnr": false_negative_rate},
        y_true=y, y_pred=y_pred, sensitive_features=group,
    )
    by = mf.by_group.copy()
    by["n"] = group.value_counts()
    by["actual_rate"] = y.groupby(group).mean()
    by = by[(by["n"] >= min_group_n) & ~by.index.isin(schema.NON_RESPONSE_VALUES)]
    by = by.sort_values("selection_rate")
    keep = group.isin(by.index)

    shap_df = pd.DataFrame(contrib, columns=features, index=df.index)
    shap_global = shap_df.abs().mean().sort_values(ascending=False)
    shap_by_group = shap_df[keep].groupby(group[keep]).mean().loc[by.index]

    attrs = [a for a in schema.AUDIT_ATTRIBUTES if a in df.columns and df[a].nunique() > 1]
    report = ModelReport(
        target_stage=target_stage, attribute=attribute, features=features, n=len(df), base_rate=base_rate,
        auc=float(roc_auc_score(y, oof)), accuracy=float(accuracy_score(y, y_pred)), by_group=by,
        dp_ratio=float(demographic_parity_ratio(y[keep], y_pred[keep], sensitive_features=group[keep])),
        dp_difference=float(demographic_parity_difference(y[keep], y_pred[keep], sensitive_features=group[keep])),
        eo_difference=float(equalized_odds_difference(y[keep], y_pred[keep], sensitive_features=group[keep])),
        shap_global=shap_global, shap_by_group=shap_by_group, proxies=proxy_scan(df, features, attrs),
    )
    if report.auc < 0.6:
        report.notes.append(
            f"The model barely predicts outcomes (AUC {report.auc:.2f}). The features explain little of who "
            "moved on, so treat group gaps here with extra caution."
        )
    if mitigate:
        report.mitigation = _threshold_optimizer(X, y, group, keep, seed, threshold, y_pred)
    return report


def _threshold_optimizer(X, y, group, keep, seed, threshold, y_pred) -> dict | None:
    """Compare against Fairlearn's ThresholdOptimizer (demographic parity), cross-fitted."""
    import lightgbm as lgb
    from fairlearn.postprocessing import ThresholdOptimizer

    Xk, yk, gk = X[keep], y[keep], group[keep]
    pred = np.zeros(len(Xk), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        for tr, te in skf.split(Xk, yk):
            to = ThresholdOptimizer(estimator=lgb.LGBMClassifier(random_state=seed, **LGB_PARAMS),
                                    constraints="demographic_parity", objective="accuracy_score",
                                    predict_method="predict_proba")
            to.fit(Xk.iloc[tr], yk.iloc[tr], sensitive_features=gk.iloc[tr])
            pred[te] = to.predict(Xk.iloc[te], sensitive_features=gk.iloc[te], random_state=seed)
    except Exception as e:  # noqa: BLE001 - diagnostics only; report rather than crash
        return {"error": str(e)}
    return {
        "dp_ratio_before": float(demographic_parity_ratio(yk, y_pred[keep], sensitive_features=gk)),
        "dp_ratio_after": float(demographic_parity_ratio(yk, pred, sensitive_features=gk)),
        "accuracy_before": float(accuracy_score(yk, y_pred[keep])),
        "accuracy_after": float(accuracy_score(yk, pred)),
    }


def summary_lines(r: ModelReport) -> list[str]:
    """Plain-language findings for the app and the report."""
    lines = []
    top = r.shap_global.index[0]
    lines.append(f"The strongest driver of past {r.target_stage} decisions was **{top.replace('_', ' ')}**.")
    strong = [(f, a, v) for f in r.proxies.index for a, v in r.proxies.loc[f].items() if v >= 0.15]
    for f, a, v in sorted(strong, key=lambda t: -t[2]):
        lines.append(
            f"**{f.replace('_', ' ')}** is a {proxy_level(v)} proxy for {a.replace('_', '/')} "
            f"(association {v:.2f}). A model can rebuild group differences through it even though "
            f"{a.replace('_', '/')} is never an input."
        )
    b = r.by_group[r.by_group["n"] >= STABLE_GROUP_N]
    if len(b) < 2:
        b = r.by_group
    hist_ratio = b["actual_rate"].min() / b["actual_rate"].max()
    model_ratio = b["selection_rate"].min() / b["selection_rate"].max()
    size_note = f" (groups with {STABLE_GROUP_N}+ applicants)" if len(b) < len(r.by_group) else ""
    lines.append(
        f"Lowest ÷ highest group rate{size_note}: past decisions **{hist_ratio:.2f}**, model **{model_ratio:.2f}**"
        + (": the model **widened** the gap." if model_ratio < hist_ratio - 0.02 else
           ": the model narrowed the gap." if model_ratio > hist_ratio + 0.02 else ": about the same.")
    )
    up = b[b["selection_rate"] - b["actual_rate"] > 0.02]
    down = b[b["actual_rate"] - b["selection_rate"] > 0.02]
    if len(up) and len(down):
        fmt = lambda d: ", ".join(f"{g} ({row.actual_rate:.0%} → {row.selection_rate:.0%})" for g, row in d.iterrows())  # noqa: E731
        lines.append(f"Compared with past decisions, the model moved **more** of {fmt(up)} forward, and "
                     f"**fewer** of {fmt(down)}.")
    return lines
