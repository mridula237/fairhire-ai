import numpy as np
import pandas as pd
import pytest

from fairhire.make_demo_data import GeneratorConfig, generate
from fairhire.model_fairness import (
    GuardError,
    correlation_ratio,
    cramers_v,
    proxy_scan,
    run_model_check,
    summary_lines,
)


@pytest.fixture(scope="module")
def proxy_report():
    df = generate(GeneratorConfig(n=4000, seed=11, referral_proxy=True))
    return run_model_check(df, "race_ethnicity", "screened")


def test_cramers_v_bounds():
    a = pd.Series(list("ab") * 200)
    assert cramers_v(a, a) == pytest.approx(1.0, abs=0.01)
    rng = np.random.default_rng(0)
    x = pd.Series(rng.choice(list("abc"), 5000))
    y = pd.Series(rng.choice(list("xyz"), 5000))
    assert cramers_v(x, y) < 0.03


def test_correlation_ratio():
    g = pd.Series(["a"] * 100 + ["b"] * 100)
    assert correlation_ratio(g, pd.Series([1.0] * 100 + [5.0] * 100)) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    assert correlation_ratio(pd.Series(rng.choice(list("ab"), 4000)), pd.Series(rng.normal(size=4000))) < 0.05


def test_guards_refuse_small_or_thin_data():
    small = generate(GeneratorConfig(n=150, seed=1))
    with pytest.raises(GuardError):
        run_model_check(small)


def test_protected_attributes_never_features():
    df = generate(GeneratorConfig(n=600, seed=2))
    with pytest.raises(ValueError, match="Protected"):
        run_model_check(df, features=["source", "gender"])


def test_proxy_is_found_and_explained(proxy_report):
    r = proxy_report
    assert r.shap_global.index[0] == "source"                     # referrals drive past screening
    assert r.proxies.loc["source", "race_ethnicity"] >= 0.15      # source carries race
    assert r.proxies.loc["education_level", "race_ethnicity"] < 0.05
    assert r.shap_by_group.loc["White", "source"] > 0 > r.shap_by_group.loc["Hispanic or Latino", "source"]
    assert r.by_group["selection_rate"].idxmax() == "White"
    assert r.dp_ratio < 0.8                                       # model fails four-fifths
    assert 0.5 <= r.auc <= 1.0


def test_mitigation_improves_parity(proxy_report):
    m = proxy_report.mitigation
    assert "error" not in m, m
    assert m["dp_ratio_after"] > m["dp_ratio_before"]
    assert m["accuracy_after"] > m["accuracy_before"] - 0.05      # little or no accuracy cost here


def test_no_proxy_without_referral_bias():
    df = generate(GeneratorConfig(n=4000, seed=3))
    ps = proxy_scan(df, ["source", "education_level", "years_experience"], ["race_ethnicity", "gender"])
    assert (ps < 0.05).all().all()


def test_summary_mentions_proxy(proxy_report):
    text = " ".join(summary_lines(proxy_report))
    assert "source" in text and "proxy" in text and "past decisions" in text
