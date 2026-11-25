"""Plotly chart builders for the FairHire app.

Color rules (validated with the dataviz palette checker):
* Groups get categorical colors in a FIXED order tied to the group, never to rank, so a
  group keeps its color when filters change. Non-response groups are never charted.
* The heat map is diverging around AIR = 1.0 (red below, gray at parity, blue above), and
  every cell carries its number; flagged cells add a warning marker, so color is never the
  only signal. The results table is the full text view of the same numbers.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from fairhire import schema

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = [[0.0, "#c43c3c"], [0.25, "#ec8a86"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
AIR_MIN, AIR_MAX = 0.4, 1.6  # color clamp; 1.0 sits exactly in the middle

STAGE_LABEL = {
    "screened": "Applied → Screened",
    "interviewed": "Screened → Interviewed",
    "offered": "Interviewed → Offered",
    "hired": "Offered → Hired",
}

_FIXED_LISTS = [schema.RACE_ETHNICITY_VALUES, schema.GENDER_VALUES, schema.AGE_BAND_VALUES, schema.SES_VALUES]
_FIXED_SLOT = {
    g: i
    for values in _FIXED_LISTS
    for i, g in enumerate(v for v in values if v not in schema.NON_RESPONSE_VALUES)
}
MAX_SERIES = 8


def group_colors(groups: list[str]) -> dict[str, str]:
    """Stable color per group. Schema labels always get the same slot (their position in their own
    category list), whatever other groups are present. Other labels (e.g. intersectional) are
    colored alphabetically after the fixed ones."""
    out = {g: CATEGORICAL[_FIXED_SLOT[g] % len(CATEGORICAL)] for g in groups if g in _FIXED_SLOT}
    rest = sorted(g for g in groups if g not in _FIXED_SLOT and g not in schema.NON_RESPONSE_VALUES)
    for i, g in enumerate(rest):
        out[g] = CATEGORICAL[i % len(CATEGORICAL)]
    return out


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 16}},
        height=height,
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "size": 13},
        hoverlabel={"font_size": 13},
        legend={"orientation": "h", "y": -0.18, "x": 0, "title": None},
    )
    return fig


def chartable_groups(full: pd.DataFrame, attribute: str) -> list[str]:
    """Groups worth charting: known response, eligible at the first stage; largest first, max 8."""
    first = full[full["stage"] == schema.STAGES[1]]
    ok = first[first["eligible"]].sort_values("n", ascending=False)
    return ok[attribute].tolist()[:MAX_SERIES]


def funnel_chart(full: pd.DataFrame, attribute: str, groups: list[str] | None = None) -> go.Figure:
    """Grouped bars: selection rate at each step, per group, with Wilson 95% CI whiskers."""
    groups = groups or chartable_groups(full, attribute)
    colors = group_colors(groups)
    fig = go.Figure()
    for g in groups:
        d = full[(full[attribute] == g) & full["stage"].isin(STAGE_LABEL)].copy()
        d["x"] = d["stage"].map(STAGE_LABEL)
        d.loc[d["n"] < schema.MIN_GROUP_N, ["rate", "rate_ci_low", "rate_ci_high"]] = float("nan")  # too few
        fig.add_bar(
            name=g,
            x=d["x"],
            y=d["rate"],
            marker={"color": colors[g], "cornerradius": 4},
            error_y={"type": "data", "symmetric": False, "array": d["rate_ci_high"] - d["rate"],
                     "arrayminus": d["rate"] - d["rate_ci_low"], "color": MUTED, "thickness": 1, "width": 3},
            customdata=d[["n", "selected", "rate_ci_low", "rate_ci_high"]].to_numpy(),
            hovertemplate=(
                f"<b>{g}</b><br>%{{x}}<br>Rate %{{y:.1%}} (95% CI %{{customdata[2]:.1%}}–%{{customdata[3]:.1%}})"
                "<br>%{customdata[1]:,} of %{customdata[0]:,}<extra></extra>"
            ),
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID, zeroline=False, title=None)
    fig.update_xaxes(title=None, showgrid=False)
    return _base_layout(fig, "Selection rate at each step, by group")


def _cell_text(row: pd.Series) -> str:
    if row["verdict"] == "no verdict" and not row.get("is_reference", False):
        return f"n={int(row['n'])}" if row["n"] < schema.MIN_GROUP_N else ""
    if row.get("is_reference", False):
        return "ref"
    mark = " ⚠" if row["verdict"] == "adverse impact indicated" else (
        " ·" if row["verdict"] == "significant gap above 4/5" else "")
    return f"{row['air']:.2f}{mark}"


def air_heatmap(full: pd.DataFrame, attribute: str) -> go.Figure:
    """Groups x stages; color = AIR (diverging at 1.0); text = AIR + marker for flagged cells.
    Cells with no verdict (n < 30) are left uncolored and show their n."""
    d = full[~full[attribute].isin(schema.NON_RESPONSE_VALUES) & full["stage"].isin(STAGE_LABEL)].copy()
    order = (d[d["stage"] == schema.STAGES[1]].sort_values("n", ascending=False)[attribute].tolist())
    order += [g for g in d[attribute].unique() if g not in order]
    stages = [s for s in STAGE_LABEL if s in set(d["stage"])]

    z, text, hover = [], [], []
    for g in order:
        zr, tr, hr = [], [], []
        for s in stages:
            row = d[(d[attribute] == g) & (d["stage"] == s)]
            if row.empty:
                zr.append(None)
                tr.append("")
                hr.append("no applicants at this step")
                continue
            r = row.iloc[0]
            show = r["eligible"] and not math.isnan(r["air"])
            zr.append(min(max(r["air"], AIR_MIN), AIR_MAX) if show else None)
            tr.append(_cell_text(r))
            p = "" if math.isnan(r["p_adjusted"]) else f"<br>adjusted p = {r['p_adjusted']:.2g}"
            hr.append(
                f"<b>{g}</b><br>{STAGE_LABEL[s]}<br>AIR {r['air']:.2f}{p}"
                f"<br>{int(r['selected'])} of {int(r['n'])} ({r['rate']:.1%})<br>{r['verdict']}"
            )
        z.append(zr)
        text.append(tr)
        hover.append(hr)

    fig = go.Figure(
        go.Heatmap(
            z=z, x=[STAGE_LABEL[s] for s in stages], y=order, text=text, customdata=hover,
            texttemplate="%{text}", textfont={"color": INK, "size": 13},
            hovertemplate="%{customdata}<extra></extra>",
            colorscale=DIVERGING, zmin=AIR_MIN, zmax=AIR_MAX, zmid=1.0, xgap=2, ygap=2,
            colorbar={"title": {"text": "AIR", "side": "top"}, "tickvals": [0.4, 0.8, 1.0, 1.6],
                      "ticktext": ["≤0.4", "0.8 (4/5)", "1.0", "≥1.6"], "len": 0.8, "thickness": 12},
        )
    )
    fig.update_yaxes(autorange="reversed", title=None, showgrid=False)
    fig.update_xaxes(side="top", title=None, showgrid=False)
    h = max(260, 46 * len(order) + 120)
    fig = _base_layout(fig, "Adverse impact ratio by group and step  (⚠ = adverse impact indicated)", h)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", margin={"t": 90})
    return fig


# ---------------------------------------------------------------------------
# Module 3 charts
# ---------------------------------------------------------------------------

HISTORY_GRAY = "#9a9890"


def model_vs_history_chart(by_group: pd.DataFrame, title: str) -> go.Figure:
    """Per group: past selection rate (gray) vs the model's selection rate (blue)."""
    d = by_group.sort_values("actual_rate")
    fig = go.Figure()
    fig.add_bar(name="Past decisions", y=d.index, x=d["actual_rate"], orientation="h",
                marker={"color": HISTORY_GRAY, "cornerradius": 4},
                hovertemplate="<b>%{y}</b><br>Past rate %{x:.1%}<extra></extra>")
    fig.add_bar(name="Model trained on them", y=d.index, x=d["selection_rate"], orientation="h",
                marker={"color": CATEGORICAL[0], "cornerradius": 4},
                customdata=d[["n"]].to_numpy(),
                hovertemplate="<b>%{y}</b><br>Model rate %{x:.1%}<br>n = %{customdata[0]:,}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_xaxes(tickformat=".0%", range=[0, 1], gridcolor=GRID, title=None)
    fig.update_yaxes(title=None)
    return _base_layout(fig, title, height=max(300, 52 * len(d) + 140))


def shap_global_chart(shap_global: pd.Series) -> go.Figure:
    """Which features drove the model's decisions (mean |SHAP|), one series, largest on top."""
    s = shap_global.sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=[i.replace("_", " ") for i in s.index], orientation="h",
                           marker={"color": CATEGORICAL[0], "cornerradius": 4},
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(gridcolor=GRID, title="Average impact on the model's score (mean |SHAP|, log-odds)")
    fig.update_yaxes(title=None)
    return _base_layout(fig, "What the model learned from past decisions", height=max(240, 50 * len(s) + 130))


def shap_by_group_chart(shap_by_group: pd.DataFrame, feature: str) -> go.Figure:
    """For one feature: average push up (blue) or down (red) on each group's score."""
    s = shap_by_group[feature].sort_values()
    colors = [DIVERGING[0][1] if v < 0 else DIVERGING[-1][1] for v in s.values]
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker={"color": colors, "cornerradius": 4},
                           hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title="Average push on the score (SHAP, log-odds)")
    fig.update_yaxes(title=None)
    return _base_layout(fig, f"How “{feature.replace('_', ' ')}” pushed each group's score",
                        height=max(260, 48 * len(s) + 130))
