"""FairHire AI v0.1: upload a hiring file (or use demo data) and get a funnel disparity audit."""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fairhire import __version__, schema  # noqa: E402
from fairhire.charts import (  # noqa: E402
    air_heatmap,
    funnel_chart,
    model_vs_history_chart,
    shap_by_group_chart,
    shap_global_chart,
)
from fairhire.fairhire_stats import add_intersection, audit_funnel, audit_stage  # noqa: E402
from fairhire.ingest import (  # noqa: E402
    apply_mapping,
    guess_stage,
    suggest_mapping,
    suggest_stage_flags,
    unmapped_stage_values,
    validate,
)
from fairhire.jd_audit import audit_jd, highlighted_html, read_docx, report_markdown  # noqa: E402
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
st.caption(f"v{__version__} · Open-source hiring equity audit · Built with the Chicago Education "
           "Advocacy Cooperative (ChiEAC)")
MODULES = ["Hiring funnel audit", "Job description audit", "Model fairness check"]
module = st.radio("Choose a check", MODULES, horizontal=True, label_visibility="collapsed")

EXAMPLE_JD = """Data Analyst (example posting, fictional)

We're a young, energetic team looking for a rockstar Data Analyst who is competitive, driven and ready to \
dominate our reporting. You'll lead analytics for our programs.

What you'll do
- Build dashboards and own our KPIs
- Work hard, play hard in a fast-paced, high-pressure environment
- Be a culture fit with our hard-charging team

What you need
- Bachelor's degree required
- Must be a digital native
- Native English speaker
- Must be able to lift 25 lbs
- Valid driver's license required

He will report to the Director of Operations."""


def render_jd_audit() -> None:
    st.markdown("Check a job posting for **gender-coded words**, **age-coded and exclusionary phrases**, "
                "**reading level**, and a **missing pay range**, and get suggested rewrites.")
    src_col, up_col = st.columns([2, 1])
    with up_col:
        jd_file = st.file_uploader("…or upload a posting (.txt or .docx)", type=["txt", "docx"])
        if st.button("Try an example posting"):
            st.session_state["jd_text"] = EXAMPLE_JD
    if jd_file is not None and st.session_state.get("jd_file_id") != jd_file.file_id:
        data = jd_file.getvalue()
        st.session_state["jd_file_id"] = jd_file.file_id  # load once per file, so later edits stick
        st.session_state["jd_text"] = (read_docx(data) if jd_file.name.lower().endswith(".docx")
                                       else data.decode("utf-8", errors="replace"))
    with src_col:
        text = st.text_area("Paste the job description", key="jd_text", height=260,
                            placeholder="Paste the full posting: title, responsibilities, requirements, pay.")
    if not text or len(text.split()) < 20:
        st.info("Paste or upload a posting (at least a few sentences) to see results.")
        return

    r = audit_jd(text)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("JD Equity Score", f"{r.score}/100")
    coding = r.gender_coding
    m2.metric("Gender coding", coding.replace("strongly ", "").replace("-coded", "").capitalize(),
              delta="strongly coded" if coding.startswith("strongly") else None, delta_color="off")
    m3.metric("Reading level", f"Grade {r.readability['fk_grade']}")
    m4.metric("Pay range listed", "Yes" if r.has_pay_range else "No")

    if r.deductions:
        st.markdown("**Why points were taken off**")
        for d, p in r.deductions:
            st.markdown(f"- {d} **(−{p})**")
    else:
        st.success("No issues found. Nice posting.")
    if not r.has_pay_range:
        st.warning("Illinois employers with 15+ employees must include the pay scale and a general description "
                   "of benefits in job postings (since Jan 1, 2025).")

    left, right = st.columns(2)
    with left:
        st.subheader("Your posting")
        st.caption("Hover a highlight to see why. 🟦 masculine-coded · 🟪 feminine-coded · 🟥 high · 🟧 medium · "
                   "⬜ low severity")
        st.html(highlighted_html(r))
    with right:
        st.subheader("Suggested version")
        st.caption("Safe word swaps are applied automatically. Flags that need judgment are listed below. "
                   "Edit freely, then download.")
        new_text = st.text_area("Suggested version", r.rewrite, height=420, label_visibility="collapsed")
        st.download_button("Download suggested version (.txt)", new_text.encode(), file_name="job_posting_revised.txt")

    if r.flags:
        st.subheader(f"Flagged phrases ({len(r.flags)})")
        st.dataframe(
            pd.DataFrame([{"Phrase": h.text, "Category": h.category, "Severity": h.severity, "Why": h.why,
                           "Suggestion": h.suggestion} for h in r.flags]),
            hide_index=True, width="stretch",
        )
    st.markdown(f"**Gender-coded words.** Masculine: {', '.join(sorted(set(r.masculine))) or 'none'} · "
                f"Feminine: {', '.join(sorted(set(r.feminine))) or 'none'}")
    st.caption("Word lists from the Gender Decoder (Matfield), based on Gaucher, Friesen & Kay (2011). Research "
               "found masculine-coded ads made jobs less appealing to women. Coding is about the overall mix, "
               "not any single word.")
    st.download_button("Download full report (.md)", report_markdown(r).encode(), file_name="jd_audit_report.md")
    st.divider()
    st.caption("Lead developer: Mridula Kalaiselvan · MIT License · " + DISCLAIMER)


@st.cache_data(show_spinner="Training the diagnostic model (about 10 seconds)…")
def cached_model_check(df: pd.DataFrame, attribute: str, target: str, features: tuple[str, ...]):
    from fairhire.model_fairness import run_model_check

    return run_model_check(df, attribute, target, list(features))


def render_model_check() -> None:
    from fairhire.model_fairness import (
        CANDIDATE_FEATURES,
        DEFAULT_FEATURES,
        LABEL,
        GuardError,
        proxy_level,
        summary_lines,
    )

    st.warning(f"⚠️ {LABEL}")
    st.markdown("Train a model on **past hiring decisions** (without any demographic data) and see what it "
                "learned: which factors drove decisions, whether any factor stands in for race or gender, and "
                "whether the model would treat groups differently.")
    sources = ["Demo data (referrals favor one group)"]
    if "clean_df" in st.session_state:
        sources.insert(0, "The file I uploaded in Hiring funnel audit")
    src = st.radio("Data", sources, horizontal=True)
    if src.startswith("Demo"):
        data = generate(GeneratorConfig(n=5000, seed=42, referral_proxy=True))
        st.caption("Demo: White applicants get referrals 5× as often, and past screeners favored referrals. "
                   "Race is never a model input. Can the check find the proxy?")
    else:
        data = st.session_state["clean_df"]

    c1, c2, c3 = st.columns(3)
    attr_opts = {k: v for k, v in ATTRIBUTES.items() if v in data.columns and v != "gender_x_race_ethnicity"
                 and data[v].nunique() > 1 and not (data[v] == "Unknown").all()}
    attr_label = c1.selectbox("Check fairness for", list(attr_opts))
    target = c2.selectbox("Decision to model", schema.STAGES[1:], format_func=lambda s: f"Reached '{s}'")
    usable = [f for f in CANDIDATE_FEATURES if f in data.columns and data[f].notna().any()
              and data[f].astype(str).nunique() > 1 and not (data[f].astype(str) == "Unknown").all()]
    feats = c3.multiselect("Model inputs (non-demographic only)", usable,
                           default=[f for f in DEFAULT_FEATURES if f in usable])
    if not feats:
        st.info("Pick at least one model input.")
        return
    try:
        r = cached_model_check(data, attr_opts[attr_label], target, tuple(feats))
    except GuardError as e:
        st.error(f"Not enough data for a fair diagnosis: {e}")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Applicants", f"{r.n:,}")
    m2.metric("Model accuracy (AUC)", f"{r.auc:.2f}", help="0.5 = coin flip, 1.0 = perfect. Low means the "
              "inputs explain little of past decisions.")
    m3.metric("Group selection ratio", f"{r.dp_ratio:.2f}", help="Lowest group's model selection rate ÷ highest "
              "(demographic parity ratio). Below 0.80 fails the four-fifths rule.")
    m4.metric("Equalized odds gap", f"{r.eo_difference:.2f}", help="Largest difference between groups in "
              "true- or false-positive rate. 0 = equal error rates.")

    st.subheader("What we found")
    for line in summary_lines(r):
        st.markdown(f"- {line}")
    for n in r.notes:
        st.info(n)

    st.plotly_chart(model_vs_history_chart(r.by_group, f"Who reached '{target}': past decisions vs the model"),
                    width="stretch")
    st.caption("Groups under 30 applicants and unknown/declined responses aren't shown. The model scores every "
               "applicant with 5-fold cross-validation, so no one is scored by a model that saw them.")
    left, right = st.columns(2)
    left.plotly_chart(shap_global_chart(r.shap_global), width="stretch")
    right.plotly_chart(shap_by_group_chart(r.shap_by_group, r.shap_global.index[0]), width="stretch")

    st.subheader("Proxy scan")
    st.caption("How strongly each model input is linked to each demographic attribute (Cramér's V or correlation "
               "ratio, 0–1). A strong link means the input can carry demographic information into a model.")
    prox = r.proxies.copy()
    labels = {v: k for k, v in ATTRIBUTES.items()}
    prox.columns = [labels.get(c, c) for c in prox.columns]
    prox.index = [i.replace("_", " ") for i in prox.index]
    st.dataframe(prox.map(lambda v: f"{v:.2f} · {proxy_level(v)}"), width="stretch")

    if r.mitigation and "error" not in r.mitigation:
        mm = r.mitigation
        st.subheader("Could a fairness constraint fix it?")
        st.markdown(f"Fairlearn's **ThresholdOptimizer** (demographic parity) changes the group selection ratio "
                    f"from **{mm['dp_ratio_before']:.2f} → {mm['dp_ratio_after']:.2f}**, with accuracy "
                    f"**{mm['accuracy_before']:.1%} → {mm['accuracy_after']:.1%}**.")
        st.caption("This is a what-if, not a recommendation to deploy a model. The better fix is upstream: change "
                   "the practice the model learned from (e.g. how referrals are weighted in screening).")

    with st.expander("Full group table"):
        st.dataframe(r.by_group.rename(columns={"actual_rate": "past_rate"}).round(3), width="stretch")
    st.divider()
    st.caption("Lead developer: Mridula Kalaiselvan · MIT License · " + DISCLAIMER)


if module == MODULES[1]:
    render_jd_audit()
    st.stop()
if module == MODULES[2]:
    render_model_check()
    st.stop()

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
if not is_demo:
    st.session_state["clean_df"] = df  # lets the Model fairness check reuse this upload

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
