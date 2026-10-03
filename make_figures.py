"""Static PNGs for the README, drawn from outputs/*.csv -> assets/."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

dep = pd.read_csv("outputs/03_theme_by_department.csv")
dep = dep.pivot(index="department", columns="theme", values=dep.columns[-1]).rename(columns=NAMES)
dep = dep[[NAMES[t] for t in ["fit", "style", "fabric", "photo", "price", "build"]]]
fig, ax = plt.subplots(figsize=(8, 4))
ax.grid(False)
ax.imshow(dep, cmap=LinearSegmentedColormap.from_list("a", ["#F5F5F5", ACCENT]), aspect="auto")
ax.set_xticks(range(dep.shape[1]), dep.columns, rotation=20, ha="right")
ax.set_yticks(range(dep.shape[0]), dep.index)
for (i, j), v in np.ndenumerate(dep.values):
    ax.annotate(f"{v:.0%}", (j, i), ha="center", va="center", fontsize=9)
save(fig, "Fit is the top complaint in every department", "", "fig4_department")
