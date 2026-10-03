"""Static PNGs for the README, drawn from outputs/*.csv -> assets/."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT, GREY, LIGHT, TEXT = "#D55E00", "#9E9E9E", "#D9D9D9", "#4D4D4D"
NAMES = {"fit": "Fit / sizing", "fabric": "Fabric / material", "build": "Construction / durability",
         "photo": "Looks different from photo", "style": "Unflattering style", "price": "Price / value",
         "tone_mixed": "Mixed tone", "tone_neg": "Negative tone"}
plt.rcParams.update({"font.family": "sans-serif", "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
                     "ytick.color": TEXT, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": LIGHT, "axes.axisbelow": True})


def save(fig, title, xlabel, name):
    fig.axes[0].set(xlabel=xlabel)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"assets/{name}.png", dpi=150)
    plt.close(fig)


def bars(d, x, label, title, xlabel, name, fmt, lo=None, hi=None):
    """Horizontal bars, top bar (last row after sorting) in accent colour."""
    d = d.sort_values(x)
    fig, ax = plt.subplots(figsize=(8, 4))
    err = [d[x] - d[lo], d[hi] - d[x]] if lo else None
    ax.barh(d[label], d[x], xerr=err, color=[GREY] * (len(d) - 1) + [ACCENT], ecolor=TEXT, capsize=3)
    edge = (np.where(d[x] >= 0, d[hi], d[lo]) if lo else d[x])
    for i, (v, e) in enumerate(zip(d[x], edge)):
        ax.annotate(fmt.format(v), (e, i), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9)
    ax.margins(x=0.15)
    ax.grid(axis="y", visible=False)
    save(fig, title, xlabel, name)


rates = pd.read_csv("outputs/01_theme_rates.csv").query("theme != 'any complaint'")
rates = rates.assign(label=rates["theme"].map(NAMES))
bars(rates, "rate", "label", "Fit is the top complaint, in a third of reviews", "Reviews mentioning theme (share, 95% CI)",
     "fig1_prevalence", "{:.1%}", "ci_low", "ci_high")

mod = pd.read_csv("outputs/rating_model.csv").query("spec == 'themes only'")
mod = mod.assign(label=mod["theme"].map(NAMES), lost=-mod["stars_lost_per_review"], drop=-mod["coef"])
bars(mod.assign(coef=mod["drop"], lo=-mod["ci_high"], hi=-mod["ci_low"]), "coef", "label",
     "Unflattering style pulls a rating down the most per mention", "Stars lost when theme present (95% CI)",
     "fig2_penalty", "{:.2f}", "lo", "hi")
bars(mod, "lost", "label", "Fit and style make up most of the stars complaints take away",
     "Stars lost per review (penalty x prevalence)", "fig3_stars_lost", "{:.2f}")

orr = pd.read_csv("outputs/08_model_odds_ratios.csv")
orr = orr[(orr["spec"] == "flags only")].assign(label=lambda d: d["term"].map(NAMES)).sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(orr["odds_ratio"], orr["label"], xerr=[orr["odds_ratio"] - orr["ci_low"], orr["ci_high"] - orr["odds_ratio"]],
            fmt="none", ecolor=TEXT, capsize=3)
ax.scatter(orr["odds_ratio"], orr["label"], s=60, zorder=3, color=[GREY] * (len(orr) - 1) + [ACCENT])
for i, v in enumerate(orr["odds_ratio"]):
    ax.annotate(f"{v:.1f}", (v, i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
ax.axvline(1, color=TEXT, ls="--")
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
ax.grid(axis="y", visible=False)
save(fig, "Style and fabric complaints most strongly predict a 1-2 star rating",
     "Odds ratio of a 1-2 star rating (log scale; 1 = no association; 95% CI)", "fig4_odds_ratios")

cal = pd.read_csv("outputs/07_model_calibration.csv")
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, .6], [0, .6], color=GREY, ls="--", label="Perfect calibration")
ax.scatter(cal["predicted"], cal["observed"], s=cal["n"], color=ACCENT, zorder=3, label="Decile of predicted risk")
ax.set(ylabel="Observed share of 1-2 stars")
ax.xaxis.set_major_formatter("{x:.0%}")
ax.yaxis.set_major_formatter("{x:.0%}")
ax.legend(frameon=False)
save(fig, "Predicted low-rating risks match what was observed", "Predicted probability of 1-2 stars (out-of-fold)",
     "fig5_calibration")
