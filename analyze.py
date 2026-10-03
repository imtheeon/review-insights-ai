"""Run sql/0*.sql -> outputs/<stem>.csv, then fit the rating-cost model -> outputs/rating_model.csv."""
import glob
import os
import duckdb
import pandas as pd
import statsmodels.formula.api as smf

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
os.makedirs("outputs", exist_ok=True)

for path in sorted(glob.glob("sql/0*.sql")):
    stem = os.path.basename(path)[:-4]
    out = duckdb.sql(open(path, encoding="utf-8").read()).df()
    out.to_csv(f"outputs/{stem}.csv", index=False)
    print(f"\n== {stem} ==\n{out.to_string(index=False)}")
    if stem == "01_theme_rates":
        n_rows = len(pd.read_parquet("data/clean/labels.parquet"))
        assert out.loc[out.theme == "any complaint", "n_sample"].iloc[0] == n_rows

# OLS rating ~ themes, HC3 robust SEs. Coef = stars per theme holding the other themes fixed.
themes = ["fit", "fabric", "build", "photo", "style", "price"]
d = pd.read_parquet("data/clean/labels.parquet")
d[themes] = d[themes].astype(int)
rows = []
for spec, extra in [("themes only", ""), ("with department", " + C(department)")]:
    m = smf.ols("rating ~ " + " + ".join(themes) + extra, d).fit(cov_type="HC3")
    ci = m.conf_int()
    for t in themes:
        prev = d[t].mean()
        rows.append(dict(spec=spec, theme=t, coef=m.params[t], ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                         p=m.pvalues[t], prevalence=prev, stars_lost_per_review=m.params[t] * prev))
model = pd.DataFrame(rows).round(4)
model.to_csv("outputs/rating_model.csv", index=False)
print(f"\n== rating_model (n={len(d)}) ==\n{model.to_string(index=False)}")
