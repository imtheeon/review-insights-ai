"""Which complaints predict a low rating (1-2 stars)? Logistic regression on the labelled sample.
Writes outputs/06_model_cv.csv, 07_model_calibration.csv, 08_model_odds_ratios.csv."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

THEMES = ["fit", "fabric", "build", "photo", "style", "price"]
d = pd.read_parquet("data/clean/labels.parquet")
y = (d["rating"] <= 2).astype(int)
# Trend has only 11 reviews: merged into Jackets so no cell is tiny. Reference levels: tone pos, department Tops.
dept = d["department"].replace({"Trend": "Jackets"})
flags = d[THEMES].astype(int)
design = {
    "flags only": flags,
    "flags + tone + department": pd.concat([flags, pd.get_dummies(d["tone"]).drop(columns="pos").add_prefix("tone_"),
                                            pd.get_dummies(dept).drop(columns="Tops").add_prefix("dept_")], axis=1).astype(int),
}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
lr = lambda: LogisticRegression(C=np.inf, max_iter=1000)

rows, oof = [], {}
for name, X in {"majority baseline": design["flags only"], **design}.items():
    est = DummyClassifier(strategy="most_frequent") if name == "majority baseline" else lr()
    p = cross_val_predict(est, X, y, cv=cv, method="predict_proba")[:, 1]
    oof[name] = p
    f = []
    for tr, te in cv.split(X, y):
        m = (DummyClassifier(strategy="most_frequent") if name == "majority baseline" else lr()).fit(X.iloc[tr], y.iloc[tr])
        pt, pred = m.predict_proba(X.iloc[te])[:, 1], m.predict(X.iloc[te])  # predict = threshold 0.5
        f.append([roc_auc_score(y.iloc[te], pt), precision_score(y.iloc[te], pred, zero_division=0),
                  recall_score(y.iloc[te], pred), brier_score_loss(y.iloc[te], pt)])
    f = np.array(f)
    rows.append(dict(model=name, threshold=0.5, base_rate=y.mean(), n=len(y),
                     **{f"{k}_{s}": g(f[:, i]) for i, k in enumerate(["auc", "precision", "recall", "brier"])
                        for s, g in [("mean", np.mean), ("sd", lambda v: np.std(v, ddof=1))]}))
pd.DataFrame(rows).round(4).to_csv("outputs/06_model_cv.csv", index=False)
print(pd.DataFrame(rows).round(3).T)

# Calibration of the full model: out-of-fold predictions in deciles (duplicates dropped if probabilities tie)
p = oof["flags + tone + department"]
b = pd.qcut(p, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"bin": b, "pred": p, "obs": y}).groupby("bin").agg(n=("obs", "size"), predicted=("pred", "mean"),
                                                                       observed=("obs", "mean")).reset_index()
cal.round(4).to_csv("outputs/07_model_calibration.csv", index=False)
print(cal.round(3))

# Odds ratios: statsmodels Logit on all rows, same designs
out = []
for name, X in design.items():
    m = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
    ci = np.exp(m.conf_int())
    for t in X.columns:
        out.append(dict(spec=name, term=t, odds_ratio=np.exp(m.params[t]), ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1],
                        p=m.pvalues[t], n_with_term=int(X[t].sum()), n_low_with_term=int(y[X[t] == 1].sum())))
orr = pd.DataFrame(out).round(4)
orr.to_csv("outputs/08_model_odds_ratios.csv", index=False)
print(orr.to_string(index=False))
assert ((orr.ci_low < orr.odds_ratio) & (orr.odds_ratio < orr.ci_high)).all()
