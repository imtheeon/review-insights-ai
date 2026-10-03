import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ACCENT, GREY, LIGHT, TEXT = "#D55E00", "#9E9E9E", "#D9D9D9", "#4D4D4D"
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
    return df, mod


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    return (p + z**2 / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d


def layout(fig, title, x, y, **kw):
    fig.update_layout(title=dict(text=title, font_size=15, x=0), xaxis_title=x, yaxis_title=y, height=360,
                      margin=dict(t=50, b=10, l=10, r=10), plot_bgcolor="#FFFFFF",
                      font=dict(family="Inter, Arial, sans-serif", color=TEXT), **{"showlegend": False, **kw})
    fig.update_xaxes(gridcolor=LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=LIGHT)
    return fig


def hbar(d, x, y, title, xtitle, fmt, hover, lo=None, hi=None, **kw):
    """Horizontal bars; d is ordered so the last row is the top bar, which gets the accent colour."""
    colors = [GREY] * (len(d) - 1) + [ACCENT]
    err = dict(type="data", symmetric=False, array=d[hi] - d[x], arrayminus=d[x] - d[lo], color=TEXT) if lo else None
    fig = go.Figure(go.Bar(x=d[x], y=d[y] + "  <b>" + d[x].map(fmt.format) + "</b>", orientation="h", marker_color=colors,
                           error_x=err, customdata=d[[lo, hi]] if lo else None, hovertemplate=hover))
    return layout(fig, title, xtitle, "", **kw)


df, mod = load()
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
    st.info("No rows match these filters.")
    st.stop()

n = len(f)
lost = -mod.set_index("theme")["stars_lost_per_review"]
st.container(border=True).markdown(
    "**Bottom line.** Fit is the dominant complaint: one review in three (33%, ±2.4 points) says an item runs small, "
    "large, long or short. Unflattering style is a distant second at 12%, but it costs the most per mention, at about "
    "1.3 stars. Together, fit and style account for about three-quarters of the rating points that complaints take away. "
    "The best next move is better size guidance on product pages, with an unflattering-style early warning for new "
    "dresses and tops.")
k = st.columns(4)
all_n = f"vs all {len(df):,}"
k[0].metric("Reviews mentioning fit", f"{f['fit'].mean():.1%}", f"{(f['fit'].mean() - df['fit'].mean()) * 100:+.1f} pts {all_n}", delta_color="off")
k[1].metric("With any complaint", f"{f['any'].mean():.0%}", f"{(f['any'].mean() - df['any'].mean()) * 100:+.1f} pts {all_n}", delta_color="off")
k[2].metric("Average rating", f"{f['rating'].mean():.2f}", f"{f['rating'].mean() - df['rating'].mean():+.2f} {all_n}", delta_color="off")
k[3].metric("Fit + style share of stars lost", f"{(lost['fit'] + lost['style']) / lost.sum():.0%}",
            f"of {lost.sum():.2f} stars complaints take off", delta_color="off", help="Full sample; not filtered")

c1, c2 = st.columns(2)
with c1.container(border=True):
    p = pd.DataFrame({"name": [THEMES[t] for t in THEMES], "k": [f[t].sum() for t in THEMES]})
    mid, half = wilson(p["k"], n)  # Wilson interval is centred on mid, not on the raw rate
    p["rate"], p["hi"], p["lo"] = p["k"] / n, mid + half, mid - half
    p = p.sort_values("rate")
    st.plotly_chart(hbar(p, "rate", "name", f"{p.iloc[-1]['name']} is the top complaint, in {p.iloc[-1]['rate']:.0%} of reviews",
                         "Reviews mentioning theme (%)", "{:.1%}", "%{y}: %{x:.1%} (95% CI %{customdata[0]:.1%} to %{customdata[1]:.1%})<extra></extra>",
                         lo="lo", hi="hi", xaxis_tickformat=".0%"), width="stretch")
    st.caption("So what: size guidance targets the one problem that dwarfs the rest. Themes overlap, so shares do not sum to 100%.")
with c2.container(border=True):
    m = mod.sort_values("coef", ascending=False)
    st.plotly_chart(hbar(m, "coef", "name", "Style costs the most stars per mention",
                         "Rating change (stars)", "{:.2f}",
                         "%{y}: %{x:.2f} stars (95% CI %{customdata[0]:.2f} to %{customdata[1]:.2f})<extra></extra>",
                         lo="ci_low", hi="ci_high"), width="stretch")
    st.caption("So what: a style complaint is the costliest single mention. Full-sample regression controlling for overlapping complaints.")

c3, c4 = st.columns(2)
with c3.container(border=True):
    m = mod.assign(lost=-mod["stars_lost_per_review"]).sort_values("lost")
    st.plotly_chart(hbar(m, "lost", "name", "Fit and style take away most of the stars",
                         "Stars lost per review", "{:.2f}",
                         "%{y}: %{x:.3f} stars per review<extra></extra>"), width="stretch")
    st.caption("So what: rare themes cost little even when each hit is large. Full sample, not filtered.")
with c4.container(border=True):
    r = pd.DataFrame([{"name": THEMES[t], "present": f.loc[f[t], "rating"].mean(), "absent": f.loc[~f[t], "rating"].mean()}
                      for t in THEMES if f[t].any() and not f[t].all()]).sort_values("present", ascending=False)
    fig = go.Figure([go.Bar(x=r["name"], y=r["present"], name="Theme present", marker_color=ACCENT, text=r["present"].map("{:.1f}".format)),
                     go.Bar(x=r["name"], y=r["absent"], name="Theme absent", marker_color=GREY, text=r["absent"].map("{:.1f}".format))])
    fig.update_yaxes(range=[1, 5])
    fig.update_traces(hovertemplate="%{x}: %{y:.2f} stars<extra>%{fullData.name}</extra>")
    st.plotly_chart(layout(fig, "Every complaint lowers the average rating", "", "Average rating (stars)",
                           barmode="group", showlegend=True), width="stretch")
    st.caption("So what: raw averages in the selection, no control for overlapping themes. Rare themes give noisy bars.")

with st.container(border=True):
    sizes = f["department"].value_counts()
    keep = sizes[sizes >= 30].index
    if len(keep) == 0:
        st.info("No department has 30+ reviews in this selection.")
    else:
        h = f[f["department"].isin(keep)].groupby("department")[list(THEMES)].mean().rename(columns=THEMES)
        fig = go.Figure(go.Heatmap(z=h.values, x=h.columns, y=h.index, colorscale=[[0, "#F5F5F5"], [1, ACCENT]],
                                   text=h.map("{:.0%}".format).values, texttemplate="%{text}",
                                   hovertemplate="%{y}, %{x}: %{z:.1%}<extra></extra>", colorbar_tickformat=".0%"))
        st.plotly_chart(layout(fig, "Fit is the top complaint in every department", "", ""), width="stretch")
        st.caption(f"So what: fix sizing everywhere, not in one category. Departments with 30+ reviews in selection ({', '.join(keep)}).")

with st.container(border=True):
    st.subheader(f"Reviews mentioning: {THEMES[browse]}")
    b = f[f[browse]]
    st.caption(f"{len(b):,} matching reviews in selection; showing up to 50.")
    st.dataframe(b[["rating", "text"]].head(50), hide_index=True, width="stretch")
