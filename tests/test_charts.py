from fairhire.charts import air_heatmap, chartable_groups, funnel_chart, group_colors
from fairhire.fairhire_stats import add_intersection, audit_funnel
from fairhire.make_demo_data import GeneratorConfig, generate


def test_colors_follow_the_group_not_the_rank():
    a = group_colors(["White", "Asian", "Hispanic or Latino"])
    b = group_colors(["Asian", "Hispanic or Latino", "White", "Black or African American"])
    assert a["White"] == b["White"] and a["Asian"] == b["Asian"]
    assert "Unknown" not in group_colors(["Unknown", "White"])


def test_charts_build_for_every_view():
    df = add_intersection(generate(GeneratorConfig(n=3000, seed=4)))
    for attr in ("race_ethnicity", "gender", "gender_x_race_ethnicity", "age_band"):
        full = audit_funnel(df, attr, "pooled")
        assert len(chartable_groups(full, attr)) <= 8
        assert funnel_chart(full, attr).data
        heat = air_heatmap(full, attr)
        flagged = full[full["verdict"] == "adverse impact indicated"]
        text = [t for row in heat.data[0].text for t in row]
        assert sum("⚠" in t for t in text) == len(flagged)
