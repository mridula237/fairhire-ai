"""FairHire AI v0.1: upload a hiring file (or use demo data) and get a funnel disparity audit."""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fairhire import __version__, schema  # noqa: E402
from fairhire.charts import air_heatmap, funnel_chart  # noqa: E402
from fairhire.fairhire_stats import add_intersection, audit_funnel, audit_stage  # noqa: E402
from fairhire.ingest import (  # noqa: E402
    apply_mapping,
    guess_stage,
    suggest_mapping,
    suggest_stage_flags,
    unmapped_stage_values,
    validate,
)
from fairhire.make_demo_data import GeneratorConfig, generate  # noqa: E402

DISCLAIMER = (
    "FairHire AI provides statistical analysis and educational information. It is not legal advice and "
    "does not replace an independent bias audit where a law requires one."
)
NOT_IN_FILE = "— not in my file —"
FIELD_LABELS = {
    "applicant_id": "Applicant ID (anonymized)",
    "req_id": "Requisition / job ID",
    "job_title": "Job title",
    "stage_reached": "Furthest stage reached",
    "gender": "Gender",
    "race_ethnicity": "Race / ethnicity",
    "age_band": "Age band (under 40 / 40+)",
    "ses_indicator": "Socioeconomic indicator",
    "source": "Applicant source",
    "education_level": "Education level",
    "years_experience": "Years of experience",
    "application_date": "Application date",
}
ATTRIBUTES = {
    "Race / ethnicity": "race_ethnicity",
    "Gender": "gender",
    "Gender × race (intersectional)": "gender_x_race_ethnicity",
    "Age band": "age_band",
    "Socioeconomic indicator": "ses_indicator",
}
REFERENCES = {
    "Everyone else, pooled (recommended)": "pooled",
    "Highest-rate group (strict four-fifths rule)": "highest",
}

st.set_page_config(page_title="FairHire AI", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_demo() -> pd.DataFrame:
    return generate(GeneratorConfig(n=5000, seed=42))


@st.cache_data(show_spinner=False)
def read_upload(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    return pd.read_csv(buf)


@st.cache_data(show_spinner="Running the audit…")
def run_audit(df: pd.DataFrame, attribute: str, reference: str) -> pd.DataFrame:
    data = add_intersection(df) if attribute == "gender_x_race_ethnicity" else df
    return audit_funnel(data, attribute, reference)


@st.cache_data(show_spinner=False)
def chi_square_by_stage(df: pd.DataFrame, attribute: str, reference: str) -> dict[str, float]:
    data = add_intersection(df) if attribute == "gender_x_race_ethnicity" else df
    out = {}
    for s in schema.STAGES[1:]:
        try:
            out[s] = audit_stage(data, attribute, s, reference).attrs["chi_square"]["p_value"]
        except ValueError:
            out[s] = float("nan")
    return out


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("FairHire AI")
st.caption(f"v{__version__} · Open-source hiring funnel disparity audit · Built with the Chicago Education "
           "Advocacy Cooperative (ChiEAC)")
st.markdown(
    "See **where** your hiring funnel treats groups differently: at which step, for which groups, how big the "
    "gap is, and whether it's statistically meaningful."
)

# ---------------------------------------------------------------------------
# Step 1: data
# ---------------------------------------------------------------------------

st.header("1 · Your data")
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("Use demo data (5,000 synthetic applicants)", type="primary"):
        st.session_state["source"] = "demo"
    st.caption("The demo data has one planted gap: Hispanic or Latino applicants pass screening → interview "
               "at 0.65× everyone else's rate. See if the audit finds it.")
with c2:
    up = st.file_uploader("…or upload an anonymized CSV / Excel export", type=["csv", "xlsx", "xls"])
    if up is not None:
        st.session_state["source"] = "upload"
    st.caption("🔒 Nothing is stored. Your file lives only in this browser session. Remove names, emails, "
               "phone numbers, addresses and dates of birth first. [Template]"
               "(https://github.com/mridula237/fairhire-ai/blob/main/data/data_template.csv) · "
               "[Data dictionary](https://github.com/mridula237/fairhire-ai/blob/main/docs/data_dictionary.md)")

source = st.session_state.get("source")
if source is None:
    st.info(DISCLAIMER)
    st.stop()

if source == "upload":
    if up is None:
        st.stop()
    try:
        raw = read_upload(up.name, up.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't read that file: {e}")
        st.stop()
else:
    raw = load_demo()

raw.columns = [str(c) for c in raw.columns]
cols = list(raw.columns)

# ---------------------------------------------------------------------------
# Step 2: map columns
# ---------------------------------------------------------------------------

suggested = suggest_mapping(cols)
flags_suggested = suggest_stage_flags(cols)
is_demo = source == "demo"

with st.expander("2 · Match your columns", expanded=not is_demo):
    st.write("We guessed which of your columns is which. Fix anything that's wrong.")
    options = [NOT_IN_FILE, *cols]
    mapping: dict[str, str | None] = {}
    grid = st.columns(3)
    stage_mode_default = 0 if suggested["stage_reached"] or not any(flags_suggested.values()) else 1
    for i, fld in enumerate(schema.ALL_COLUMNS):
        if fld == "stage_reached":
            continue
        guess = suggested[fld]
        pick = grid[i % 3].selectbox(FIELD_LABELS[fld], options, index=options.index(guess) if guess else 0,
                                     key=f"map_{fld}")
        mapping[fld] = None if pick == NOT_IN_FILE else pick

    st.markdown("**How far did each applicant get?**")
    mode = st.radio("Stage format", ["One column with the furthest stage", "A yes/no column for each stage"],
                    index=stage_mode_default, horizontal=True, label_visibility="collapsed")
    stage_flags: dict[str, str | None] = {}
    stage_value_map: dict[str, str] = {}
    if mode.startswith("One"):
        g = suggested["stage_reached"]
        pick = st.selectbox(FIELD_LABELS["stage_reached"], options, index=options.index(g) if g else 0)
        mapping["stage_reached"] = None if pick == NOT_IN_FILE else pick
        unplaced = unmapped_stage_values(raw, mapping["stage_reached"])
        if mapping["stage_reached"]:
            placed = {v: guess_stage(v) for v in raw[mapping["stage_reached"]].dropna().astype(str).unique()}
            known = {k: v for k, v in placed.items() if v}
            if known:
                st.caption("Stage values we recognized: " + "; ".join(f"{k} → {v}" for k, v in
                                                                      sorted(known.items())[:12]))
        if unplaced:
            st.warning(f"We couldn't place {len(unplaced)} stage value(s). Tell us the furthest step each one "
                       "means:")
            sg = st.columns(3)
            for j, v in enumerate(unplaced):
                choice = sg[j % 3].selectbox(f"“{v}” means…", ["(choose)", *schema.STAGES], key=f"stage_{v}")
                if choice != "(choose)":
                    stage_value_map[v] = choice
    else:
        mapping["stage_reached"] = None
        fg = st.columns(4)
        for j, stg in enumerate(schema.STAGES[1:]):
            g = flags_suggested[stg]
            pick = fg[j].selectbox(f"Reached {stg}? (yes/no)", options, index=options.index(g) if g else 0,
                                   key=f"flag_{stg}")
            stage_flags[stg] = None if pick == NOT_IN_FILE else pick

df = apply_mapping(raw, mapping, stage_flags or None, stage_value_map)

# ---------------------------------------------------------------------------
# Step 3: check
# ---------------------------------------------------------------------------

report = validate(df, cols, mapping, stage_flags or None)
with st.expander("3 · Data check", expanded=not report.ok or bool(report.warnings and not is_demo)):
    for m in report.errors:
        st.error(m)
    for m in report.warnings:
        st.warning(m)
    for m in report.info:
        st.success(m)
if not report.ok:
    st.error("Fix the issues in step 3 to see results.")
    st.stop()

# ---------------------------------------------------------------------------
# Step 4: results
# ---------------------------------------------------------------------------

st.header("4 · Results")
available = {label: a for label, a in ATTRIBUTES.items()
             if (a == "gender_x_race_ethnicity" and mapping.get("gender") and mapping.get("race_ethnicity"))
             or (a != "gender_x_race_ethnicity" and mapping.get(a))}
f1, f2, f3 = st.columns([1.2, 1.4, 1])
attr_label = f1.selectbox("Compare groups by", list(available))
ref_label = f2.selectbox("Compare each group against", list(REFERENCES))
reqs = ["All requisitions (pooled)", *sorted(df["req_id"].unique())]
req = f3.selectbox("Requisition", reqs)
attribute, reference = available[attr_label], REFERENCES[ref_label]
data = df if req == reqs[0] else df[df["req_id"] == req]

try:
    full = run_audit(data, attribute, reference)
except ValueError as e:
    st.warning(f"Not enough data for this view: {e}. Try 'All requisitions' or another comparison.")
    st.stop()

flagged = full[full["verdict"] == "adverse impact indicated"]
watch = full[full["verdict"].isin(["inconclusive: not significant", "significant gap above 4/5"])]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Applicants", f"{len(data):,}")
m2.metric("Groups audited", int(full.loc[full["stage"] == "screened", "eligible"].sum()))
m3.metric("Adverse impact indicated", len(flagged))
m4.metric("Worth a closer look", len(watch))

st.subheader("What we found")
stage_names = {"screened": "applied → screened", "interviewed": "screened → interviewed",
               "offered": "interviewed → offered", "hired": "offered → hired"}
if flagged.empty:
    st.success("No step shows adverse impact that is both below four-fifths (0.80) and statistically "
               "significant after correcting for the number of tests.")
for _, r in flagged.sort_values("air").iterrows():
    st.error(
        f"**{r[attribute]}** at **{stage_names[r['stage']]}**: selected at {r['rate']:.0%} vs "
        f"{r['reference_rate']:.0%} for the comparison group → impact ratio **{r['air']:.2f}** "
        f"(95% CI {r['air_ci_low']:.2f}–{r['air_ci_high']:.2f}), adjusted p = {r['p_adjusted']:.2g}."
    )
def _watch_line(r: pd.Series) -> str:
    why = ("below 0.80 but not statistically significant (often small numbers)"
           if r["verdict"].startswith("inconclusive") else "above 0.80 but statistically significant")
    return f"**{r[attribute]}** at **{stage_names[r['stage']]}**: ratio {r['air']:.2f}, {why}."


watch_sorted = watch.sort_values("air")
for _, r in watch_sorted.head(3).iterrows():
    st.warning(_watch_line(r))
if len(watch_sorted) > 3:
    with st.expander(f"{len(watch_sorted) - 3} more worth a closer look"):
        for _, r in watch_sorted.iloc[3:].iterrows():
            st.markdown("- " + _watch_line(r))
small = sorted(set(full.loc[(full["note"].str.startswith("caution")), attribute]))
if small:
    st.caption("Too few applicants for a verdict (n < 30 at some step): " + ", ".join(small) + ".")

st.plotly_chart(funnel_chart(full, attribute), width="stretch")
st.caption("Bars show the share of applicants at each step who moved on; whiskers are 95% confidence "
           "intervals. Bars are hidden where a group has fewer than 30 applicants at that step; unknown/declined "
           "responses aren't charted.")
st.plotly_chart(air_heatmap(full, attribute), width="stretch")
st.caption("Each cell is a group's rate ÷ the comparison rate. Red = below parity, blue = above. "
           "⚠ = adverse impact indicated · “·” = significant gap above 0.80 · n=… = too few for a verdict.")

st.subheader("Full results")
pretty = full.rename(columns={attribute: "group"})[
    ["stage", "group", "n", "selected", "rate", "rate_ci_low", "rate_ci_high", "reference_rate", "air",
     "air_ci_low", "air_ci_high", "test", "p_value", "p_adjusted", "sd_units", "verdict", "note"]
]
pretty["stage"] = pretty["stage"].map(stage_names)
st.dataframe(pretty.round(3), width="stretch", hide_index=True)
chi = chi_square_by_stage(data, attribute, reference)
st.caption("Chi-square across groups, by step: " + " · ".join(
    f"{stage_names[s]} p = {p:.2g}" for s, p in chi.items() if p == p))
st.download_button("Download results (CSV)", pretty.to_csv(index=False).encode(),
                   file_name=f"fairhire_results_{attribute}.csv", mime="text/csv")

with st.expander("How to read this · methods"):
    st.markdown(
        "- **Selection rate** at a step = applicants who moved on ÷ applicants who reached the step before.\n"
        "- **Impact ratio (AIR)** = a group's rate ÷ the comparison rate. Under 29 CFR 1607.4(D), below "
        "**0.80** is generally treated as evidence of adverse impact.\n"
        "- **Adverse impact indicated** requires both: AIR below 0.80 **and** a statistically significant gap "
        "(Holm-adjusted p < 0.05, because the audit runs many tests at once).\n"
        "- **Comparison group:** *pooled* compares each group with everyone else combined (stable). *Highest* "
        "is the strict four-fifths rule, but a small group that happens to score high can become the benchmark "
        "and make every other group look worse.\n"
        "- Tests: two-proportion z-test, or Fisher's exact for small counts; Wilson and Katz 95% intervals.\n"
        "- Full write-up: [methods.md](https://github.com/mridula237/fairhire-ai/blob/main/docs/methods.md)"
    )

st.divider()
st.caption("Lead developer: Mridula Kalaiselvan · MIT License · " + DISCLAIMER)
