import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ORANGE, MUTED, GRAY = "#d9622b", "#e8b9a3", "#9aa5b1"
THEMES = {"fit": "Fit / sizing", "fabric": "Fabric / material", "build": "Construction / durability",
          "photo": "Looks different from photo", "style": "Unflattering style", "price": "Price / value"}
st.set_page_config(page_title="Review Insights", layout="wide")


@st.cache_data
def load():
    df = pd.read_parquet("data/clean/labels.parquet")
    df[list(THEMES)] = df[list(THEMES)].fillna(False).astype(bool)
    df["any"] = df[list(THEMES)].any(axis=1)
    mod = pd.read_csv("outputs/rating_model.csv")
    mod = mod[mod["spec"] == "themes only"].assign(name=lambda d: d["theme"].map(THEMES))
    return df, mod, pd.read_csv("outputs/05_llm_vs_reference.csv")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    return (p + z**2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d


def style(fig, x, y, **kw):
    fig.update_layout(xaxis_title=x, yaxis_title=y, margin=dict(t=20, b=10, l=10, r=10),
                      legend_title_text="", height=340, **kw)
    return fig


df, mod, chk = load()
st.title("What customers complain about, and what it costs in stars")
st.caption(f"Random sample of {len(df):,} reviews labelled by a local llama3.2 model (Kaggle Women's E-Commerce "
           "Clothing Reviews). Estimates carry sampling error; error bars are 95% intervals.")

with st.sidebar:
    st.header("Filters")
    dept = st.multiselect("Department", sorted(df["department"].dropna().unique()))
    rate_sel = st.multiselect("Rating", [1, 2, 3, 4, 5])
    browse = st.selectbox("Theme for review browser", list(THEMES), format_func=THEMES.get)

f = df
if dept:
    f = f[f["department"].isin(dept)]
if rate_sel:
    f = f[f["rating"].isin(rate_sel)]
if f.empty:
    st.warning("No reviews match these filters.")
    st.stop()

n = len(f)
rates = pd.Series({t: f[t].mean() for t in THEMES})
agree = chk.loc[chk["theme"] == "all six themes match", "agreement"].iloc[0]
k = st.columns(5)
k[0].metric("Reviews in selection", f"{n:,}")
k[1].metric("Avg rating", f"{f['rating'].mean():.2f}")
k[2].metric("With any complaint", f"{f['any'].mean() * 100:.0f}%")
k[3].metric("Top complaint", THEMES[rates.idxmax()], f"{rates.max() * 100:.0f}% of reviews", delta_color="off")
k[4].metric("LLM vs reference labels: all themes match", f"{agree * 100:.0f}%",
            help=f"Exact match on all six themes across {int(chk['n'].iloc[0])} reviews re-labelled by Claude as a reference")

c1, c2 = st.columns(2)
with c1.container(border=True):
    st.subheader("Complaint prevalence by theme")
    p = pd.DataFrame({"name": [THEMES[t] for t in THEMES], "k": [f[t].sum() for t in THEMES]})
    mid, half = wilson(p["k"], n)  # Wilson interval is centred on mid, not on the raw rate
    p["rate"], p["hi"], p["lo"] = p["k"] / n * 100, (mid + half) * 100, (mid - half) * 100
    p = p.sort_values("rate")
    fig = px.bar(p, x="rate", y="name", orientation="h", error_x=p["hi"] - p["rate"],
                 error_x_minus=p["rate"] - p["lo"], color_discrete_sequence=[ORANGE])
    fig.update_xaxes(ticksuffix="%")
    st.plotly_chart(style(fig, "Reviews mentioning theme (%)", ""), width="stretch")
    st.caption(f"Share of the {n:,} selected reviews; whiskers are Wilson 95% intervals. Themes overlap, so shares do not sum to 100%.")
with c2.container(border=True):
    st.subheader("Star penalty per theme")
    m = mod.sort_values("coef", ascending=False).assign(sig=lambda d: (d["ci_low"] > 0) | (d["ci_high"] < 0))
    fig = px.bar(m, x="coef", y="name", orientation="h", color="sig",
                 error_x=m["ci_high"] - m["coef"], error_x_minus=m["coef"] - m["ci_low"],
                 color_discrete_map={True: ORANGE, False: MUTED})
    fig.update_layout(showlegend=False)
    st.plotly_chart(style(fig, "Change in star rating when theme present (stars)", ""), width="stretch")
    st.caption("Regression on the full sample, not filtered by the sidebar; controls for overlapping complaints. "
               "Light bars: interval crosses 0, effect not distinguishable from none.")

c3, c4 = st.columns(2)
with c3.container(border=True):
    st.subheader("Where the rating points go")
    m = mod.assign(lost=-mod["stars_lost_per_review"]).sort_values("lost")
    fig = px.bar(m, x="lost", y="name", orientation="h", color_discrete_sequence=[ORANGE])
    st.plotly_chart(style(fig, "Stars lost per review (penalty x prevalence)", ""), width="stretch")
    top = m.iloc[-1]
    st.caption(f"Full sample, not filtered. {top['name']} costs the most: {top['lost']:.2f} stars per review on average. "
               "Rare themes cost little even when each hit is large.")
with c4.container(border=True):
    st.subheader("Average rating with vs without theme")
    r = pd.DataFrame([{"name": THEMES[t], "group": g, "avg": f.loc[f[t] == w, "rating"].mean(),}
                      for t in THEMES for g, w in [("Theme present", True), ("Theme absent", False)]
                      if (f[t] == w).any()])
    fig = px.bar(r, x="name", y="avg", color="group", barmode="group",
                 color_discrete_map={"Theme present": ORANGE, "Theme absent": GRAY})
    fig.update_yaxes(range=[1, 5])
    st.plotly_chart(style(fig, "Theme", "Average rating (stars)"), width="stretch")
    st.caption("Raw averages in the selection (no controls for overlapping themes). "
               "Themes with few mentions give noisy bars; see the prevalence chart.")

with st.container(border=True):
    st.subheader("Theme rate by department")
    sizes = f["department"].value_counts()
    keep = sizes[sizes >= 30].index
    if len(keep) == 0:
        st.info("No department has 30+ reviews in this selection.")
    else:
        h = f[f["department"].isin(keep)].groupby("department")[list(THEMES)].mean().mul(100).rename(columns=THEMES)
        fig = px.imshow(h, text_auto=".0f", aspect="auto", color_continuous_scale="Oranges")
        fig.update_layout(coloraxis_colorbar_title="%")
        st.plotly_chart(style(fig, "Theme", "Department"), width="stretch")
        st.caption(f"% of reviews with each theme; departments with 30+ reviews in selection ({', '.join(keep)}).")

with st.container(border=True):
    st.subheader(f"Reviews mentioning: {THEMES[browse]}")
    b = f[f[browse]]
    st.caption(f"{len(b):,} matching reviews in selection; showing up to 50.")
    st.dataframe(b[["rating", "text"]].head(50), hide_index=True, width="stretch")
