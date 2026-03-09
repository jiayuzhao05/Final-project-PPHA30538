from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = REPO_ROOT / "data" / "derived-data"


st.set_page_config(
    page_title="News Sentiment & FDIC Banking Stress (State-Year)",
    layout="wide",
)

st.title("News Sentiment & FDIC Banking Stress (State-Year)")
st.caption(
    "Presentation demo: combine a year-level regulatory-news sentiment index with FDIC state-year fundamentals "
    "to predict stress and visualize geographic concentration."
)

panel_path = DERIVED / "state_year_panel.csv"
if not panel_path.exists():
    st.error(
        "Missing derived file `data/derived-data/state_year_panel.csv`. "
        "Run `python preprocessing.py` first to build the presentation dataset."
    )
    st.stop()

df = pd.read_csv(panel_path)
df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

years = sorted([int(y) for y in df["YEAR"].dropna().unique().tolist()])
if not years:
    st.error("No valid YEAR values found in derived panel.")
    st.stop()

if "StressScore" in df.columns:
    years_with_stress = (
        df.loc[df["StressScore"].notna(), "YEAR"].dropna().astype(int).unique().tolist()
    )
else:
    years_with_stress = []
years_with_stress = sorted(years_with_stress)
default_year = max(years_with_stress) if years_with_stress else max(years)

STATE_FIPS = {
    "Alabama": 1,
    "Alaska": 2,
    "Arizona": 4,
    "Arkansas": 5,
    "California": 6,
    "Colorado": 8,
    "Connecticut": 9,
    "Delaware": 10,
    "District of Columbia": 11,
    "Florida": 12,
    "Georgia": 13,
    "Hawaii": 15,
    "Idaho": 16,
    "Illinois": 17,
    "Indiana": 18,
    "Iowa": 19,
    "Kansas": 20,
    "Kentucky": 21,
    "Louisiana": 22,
    "Maine": 23,
    "Maryland": 24,
    "Massachusetts": 25,
    "Michigan": 26,
    "Minnesota": 27,
    "Mississippi": 28,
    "Missouri": 29,
    "Montana": 30,
    "Nebraska": 31,
    "Nevada": 32,
    "New Hampshire": 33,
    "New Jersey": 34,
    "New Mexico": 35,
    "New York": 36,
    "North Carolina": 37,
    "North Dakota": 38,
    "Ohio": 39,
    "Oklahoma": 40,
    "Oregon": 41,
    "Pennsylvania": 42,
    "Rhode Island": 44,
    "South Carolina": 45,
    "South Dakota": 46,
    "Tennessee": 47,
    "Texas": 48,
    "Utah": 49,
    "Vermont": 50,
    "Virginia": 51,
    "Washington": 53,
    "West Virginia": 54,
    "Wisconsin": 55,
    "Wyoming": 56,
}

MAP_METRICS = {
    "Composite stress index": "StressScore",
    "Chance of a stress year": "p_bad_year",
    "Severity of profitability decline": "sev_hat",
    "Change in ROA (ΔROA)": "DROA",
    "Return on assets (ROA)": "ROA",
    "Share of negative regulatory news": "sent_neg_share",
    "Number of regulatory-news headlines": "news_count",
}

with st.sidebar:
    st.header("Controls")
    year = st.slider(
        "Select year",
        min_value=min(years),
        max_value=max(years),
        value=default_year,
        step=1,
    )
    x_max = st.slider(
        "Max x-axis value for news negativity (zoom in)",
        min_value=0.1,
        max_value=0.5,
        value=0.25,
        step=0.02,
        help="Most sentiment values lie between 0.05–0.15. Lower this to zoom in.",
    )
    use_sample = st.checkbox(
        "Use 1% random sample for scatter plot (faster on cloud)",
        value=False,
        help="Reduces scatter plot data to 1% for quicker loading on Streamlit Cloud.",
    )
    map_metric_label = st.selectbox(
        "Map metric",
        options=list(MAP_METRICS.keys()),
        index=0,
    )
    map_value_col = MAP_METRICS[map_metric_label]
    st.caption(
        "The composite stress index combines how often and how severely profitability deteriorates."
    )

df_y = df[df["YEAR"] == year].copy()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment vs change in bank profitability")
    plot_df = df.dropna(subset=["sent_neg_share", "DROA", "YEAR"]).copy()
    plot_df = plot_df[plot_df["YEAR"].between(year - 5, year)]
    if use_sample and len(plot_df) > 100:
        plot_df = plot_df.sample(frac=0.01, random_state=42)

    if plot_df.empty:
        st.info("No sentiment/ΔROA data available for plotting.")
    else:
        chart = (
            alt.Chart(plot_df)
            .mark_circle(size=50, opacity=0.35)
            .encode(
                x=alt.X(
                    "sent_neg_share:Q",
                    title="Share of negative regulatory news (yearly mean)",
                    scale=alt.Scale(domain=[0, x_max]),
                ),
                y=alt.Y("DROA:Q", title="Change in ROA (ΔROA)"),
                color=alt.Color("bad_year:N", title="Stress year (0/1)"),
                tooltip=[
                    alt.Tooltip("STNAME:N", title="State"),
                    alt.Tooltip("YEAR:Q", title="Year"),
                    alt.Tooltip("sent_neg_share:Q", title="Share negative news"),
                    alt.Tooltip("DROA:Q", title="ΔROA"),
                    alt.Tooltip("ROA:Q", title="ROA"),
                    alt.Tooltip("StressScore:Q", title="Composite stress index"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Yearly sentiment index (filtered headlines)")
    sent_cols = ["YEAR", "sent_mean", "sent_neg_share", "news_count"]
    sent = df[sent_cols].drop_duplicates().dropna(subset=["YEAR"]).sort_values("YEAR")
    sent = sent[sent["YEAR"].between(year - 10, year)]

    if sent.empty:
        st.info("No sentiment index available.")
    else:
        sent_melt = sent.melt(
            id_vars=["YEAR"],
            value_vars=["sent_mean", "sent_neg_share"],
            var_name="var",
            value_name="value",
        )
        sent_melt["metric"] = sent_melt["var"].map({
            "sent_mean": "Mean sentiment score (pos − neg)",
            "sent_neg_share": "Share of negative headlines",
        })
        chart2 = (
            alt.Chart(sent_melt)
            .mark_line(point=True)
            .encode(
                x=alt.X("YEAR:Q", title="Year"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Sentiment metric"),
                tooltip=[
                    alt.Tooltip("YEAR:Q", title="Year"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Value"),
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart2, use_container_width=True)

st.subheader("Geographic stress map")

map_df = df_y[["STNAME", map_value_col]].copy()
map_df["STNAME"] = map_df["STNAME"].astype(str).str.strip()
map_df["id"] = map_df["STNAME"].map(STATE_FIPS)
map_df = map_df.dropna(subset=["id"]).copy()
map_df["id"] = map_df["id"].astype(int)

us_states = alt.topo_feature(
    "https://cdn.jsdelivr.net/npm/vega-datasets@v1.29.0/data/us-10m.json",
    "states",
)

background = alt.Chart(us_states).mark_geoshape(fill="lightgrey", stroke="white")

foreground = (
    alt.Chart(us_states)
    .mark_geoshape(stroke="white")
    .transform_lookup(
        lookup="id",
        from_=alt.LookupData(map_df, "id", ["STNAME", map_value_col]),
    )
    .encode(
        color=alt.Color(
            f"{map_value_col}:Q",
            title=map_metric_label,
            scale=alt.Scale(scheme="orangered"),
        ),
        tooltip=[
            alt.Tooltip("STNAME:N", title="State"),
            alt.Tooltip(f"{map_value_col}:Q", title=map_metric_label),
        ],
    )
)

st.altair_chart(
    (background + foreground)
    .project(type="albersUsa")
    .properties(
        width=700,
        height=430,
        title=f"{map_metric_label} — {year}",
    ),
    use_container_width=True,
)

with st.expander("Show state-year table (selected year)"):
    col_labels = {
        "STNAME": "State",
        "YEAR": "Year",
        "ROA": "Return on assets",
        "DROA": "Change in ROA",
        "bad_year": "Stress year (0/1)",
        "severity": "Severity of decline",
        "p_bad_year": "Chance of stress year",
        "sev_hat": "Severity (estimated)",
        "StressScore": "Composite stress index",
        "sent_mean": "Mean sentiment score",
        "sent_neg_share": "Share of negative headlines",
        "news_count": "Headline count",
    }
    show_cols = [c for c in col_labels if c in df_y.columns]
    display_df = df_y[show_cols].sort_values("StressScore", ascending=False).copy()
    display_df = display_df.rename(columns={k: col_labels[k] for k in show_cols})
    st.dataframe(display_df, use_container_width=True)

