"""Week 5 benchmark: reproduce Fairlearn's published quickstart numbers on UCI Adult.

Why: before trusting FairHire's model-fairness numbers, check that our Fairlearn setup gives
the same answers as the Fairlearn docs on the same data and model.

Reference (Fairlearn v0.9 quickstart, https://fairlearn.org/v0.9/quickstart.html):
    DecisionTreeClassifier(min_samples_leaf=10, max_depth=4), fit and scored on all 48,842 rows
    accuracy overall 0.8443, Female 0.9251, Male 0.8042
    selection rate overall 0.1638, Female 0.0635, Male 0.2135
    ExponentiatedGradient(DemographicParity) selection rate: overall 0.1661, Female 0.1552, Male 0.1715

Run (needs internet for OpenML):  PYTHONPATH=src python scripts/benchmark_adult.py
"""

import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import DemographicParity, ExponentiatedGradient
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

PUBLISHED = {
    "accuracy overall": 0.8443, "accuracy Female": 0.9251, "accuracy Male": 0.8042,
    "selection overall": 0.1638, "selection Female": 0.0635, "selection Male": 0.2135,
    "mitigated selection overall": 0.1661, "mitigated selection Female": 0.1552,
    "mitigated selection Male": 0.1715,
}
TOL = 0.01  # tree tie-breaking can differ slightly across scikit-learn versions


def main() -> None:
    data = fetch_adult(as_frame=True)
    X = pd.get_dummies(data.data)
    y = (data.target == ">50K") * 1
    sex = data.data["sex"]

    clf = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4, random_state=0).fit(X, y)
    y_pred = clf.predict(X)
    acc = MetricFrame(metrics=accuracy_score, y_true=y, y_pred=y_pred, sensitive_features=sex)
    sr = MetricFrame(metrics=selection_rate, y_true=y, y_pred=y_pred, sensitive_features=sex)

    np.random.seed(0)
    mit = ExponentiatedGradient(DecisionTreeClassifier(min_samples_leaf=10, max_depth=4, random_state=0),
                                DemographicParity())
    mit.fit(X, y, sensitive_features=sex)
    srm = MetricFrame(metrics=selection_rate, y_true=y, y_pred=mit.predict(X), sensitive_features=sex)

    ours = {
        "accuracy overall": acc.overall, "accuracy Female": acc.by_group["Female"],
        "accuracy Male": acc.by_group["Male"],
        "selection overall": sr.overall, "selection Female": sr.by_group["Female"],
        "selection Male": sr.by_group["Male"],
        "mitigated selection overall": srm.overall, "mitigated selection Female": srm.by_group["Female"],
        "mitigated selection Male": srm.by_group["Male"],
    }
    ok = True
    print(f"{'metric':32s} {'published':>9s} {'ours':>8s}  match")
    for k, ref in PUBLISHED.items():
        match = abs(float(ours[k]) - ref) <= TOL
        ok &= match
        print(f"{k:32s} {ref:9.4f} {float(ours[k]):8.4f}  {'yes' if match else 'NO'}")
    print("\nBENCHMARK", "PASSED" if ok else "FAILED", f"(tolerance ±{TOL})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
