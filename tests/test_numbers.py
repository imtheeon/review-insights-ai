"""README numbers must match the saved outputs, and the outputs must be internally consistent."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
RATES = pd.read_csv(ROOT / "outputs" / "01_theme_rates.csv").set_index("theme")
LLM = pd.read_csv(ROOT / "outputs" / "05_llm_vs_reference.csv").set_index("theme")
POP = pd.read_csv(ROOT / "outputs" / "04_sample_vs_population.csv")
MODEL = pd.read_csv(ROOT / "outputs" / "rating_model.csv")


def shown(value, places=1):
    step = 10 ** -places
    return any(f"{value + k * step:.{places}f}" in README for k in (-1, 0, 1))


def test_rates_are_consistent_with_counts():
    for theme, r in RATES.iterrows():
        assert r["n_with_theme"] / r["n_sample"] == pytest.approx(r["rate"], abs=1e-3)
        assert r["ci_low"] <= r["rate"] <= r["ci_high"]


def test_fit_is_the_top_complaint_and_the_readme_says_so():
    themes = RATES.drop(index="any complaint")
    assert themes["rate"].idxmax() == "fit"
    assert shown(themes.loc["fit", "rate"] * 100)
    assert shown(themes.loc["style", "rate"] * 100)


def test_construction_weakness_is_disclosed():
    assert LLM.loc["build", "recall"] < 0.25
    assert "undercount" in README


def test_sample_matches_population_within_one_point():
    assert POP["diff"].abs().max() < 0.009 + 1e-9
    assert "0.9 points" in README


def test_readme_penalties_match_the_model():
    base = MODEL[MODEL["spec"] == "themes only"].set_index("theme")
    assert shown(abs(base.loc["style", "coef"]), 2)
    assert shown(abs(base.loc["fit", "coef"]), 2)
