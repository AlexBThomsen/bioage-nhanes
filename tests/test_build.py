"""Sanity checks on the processed table (run after `python -m bioage.build`)."""
from pathlib import Path

import pandas as pd
import pytest

P = Path(__file__).resolve().parents[1] / "data" / "processed" / "nhanes.parquet"
pytestmark = pytest.mark.skipif(not P.exists(), reason="run bioage.build first")


@pytest.fixture(scope="module")
def df():
    return pd.read_parquet(P)


def test_one_row_per_person(df):
    assert not df.duplicated(["cycle", "SEQN"]).any()


def test_adults_only(df):
    assert df["age"].min() >= 20


def test_units_plausible(df):
    med = df.median(numeric_only=True)
    assert 38 < med["albumin"] < 48          # g/L
    assert 50 < med["creatinine"] < 100      # umol/L
    assert 4.5 < med["glucose"] < 6.5        # mmol/L
    assert 0.05 < med["crp"] < 0.5           # mg/dL
    assert 110 < med["sbp"] < 135


def test_crp_consistent_across_assays(df):
    by = df.groupby("cycle")["crp"].median()
    assert by.max() / by.min() < 3


def test_outcomes(df):
    assert set(df["dead"].dropna().unique()) <= {0, 1}
    assert (df.loc[df["stroke_death"] == 1, "dead"] == 1).all()


def test_zeros_survive_cleaning(df):
    # regression: SAS missing-code cleanup once turned real zeros (alive, male) into NaN
    assert (df["dead"] == 0).sum() > (df["dead"] == 1).sum()
    assert (df["female"] == 0).sum() > 10000
