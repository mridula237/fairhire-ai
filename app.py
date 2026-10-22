"""FairHire AI: Streamlit entry point (week 1: hello world + demo data preview)."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fairhire import __version__, schema  # noqa: E402
from fairhire.make_demo_data import GeneratorConfig, generate  # noqa: E402

DISCLAIMER = (
    "FairHire AI provides statistical analysis and educational information. "
    "It is not legal advice and does not replace an independent bias audit where a law requires one."
)

st.set_page_config(page_title="FairHire AI", page_icon="⚖️", layout="wide")

st.title("FairHire AI")
st.caption(
    f"v{__version__} · Open-source hiring funnel disparity audit · "
    "Built with the Chicago Education Advocacy Cooperative (ChiEAC)"
)
st.info(DISCLAIMER)

st.markdown(
    """
**What this will do:** upload an anonymized applicant file (or use demo data), and FairHire AI shows
where your hiring funnel treats groups differently: at which stage, for which groups, how large the gap is,
and whether it is statistically meaningful.

**Coming soon:** 1 · Funnel Disparity Audit 2 · Job Description Language Audit
3 · Model Fairness Check 4 · Equity Assessment Report
"""
)

st.warning(
    "Privacy: never upload names, emails, phone numbers, addresses, or dates of birth. "
    "Nothing you upload is stored; data lives only in this browser session."
)


@st.cache_data
def load_demo(seed: int) -> pd.DataFrame:
    return generate(GeneratorConfig(n=5000, seed=seed))


if st.button("Load demo data (5,000 synthetic applicants)", type="primary"):
    st.session_state["df"] = load_demo(42)

if "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("Demo data preview")
    st.dataframe(df.head(20), width="stretch")

    st.subheader("Applicants reaching each stage, by race/ethnicity")
    order = {s: i for i, s in enumerate(schema.STAGES)}
    reached = df["stage_reached"].map(order)
    counts = pd.DataFrame(
        {stage: (reached >= i).groupby(df["race_ethnicity"]).sum() for i, stage in enumerate(schema.STAGES)}
    )
    st.dataframe(counts, width="stretch")
    st.caption("Full selection rates, impact ratios, and significance tests arrive in v0.1 (week 3).")

st.divider()
st.caption("Lead developer: Mridula Kalaiselvan · MIT License · " + DISCLAIMER)
