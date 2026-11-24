"""Translate a new laboratory's marker values onto the training data's scale, without outcome data.

Use case: a model trained on NHANES lab results is applied to results from another laboratory whose assays read
systematically higher or lower. Each function takes the new lab's values plus the training reference and returns
values on the training scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_BANDS = [20, 40, 60, 200]


def quantile_map(values: pd.Series, reference: pd.Series) -> pd.Series:
    """Replace each value by the reference value at the same percentile. NaNs stay NaN."""
    out = pd.Series(np.nan, index=values.index, dtype=float)
    ref = np.sort(reference.dropna().to_numpy(float))
    ok = values.notna()
    if ok.sum() == 0 or len(ref) == 0:
        return out
    # mid-rank percentiles so the smallest and largest values don't map to the reference extremes
    pct = (values[ok].rank(method="average") - 0.5) / ok.sum()
    out[ok] = np.quantile(ref, pct.to_numpy())
    return out


def strata(df: pd.DataFrame) -> pd.Series:
    band = pd.cut(df["age"], AGE_BANDS, right=False, labels=False)
    return df["female"].astype(int).astype(str) + "_" + band.astype(str)


def map_markers(new: pd.DataFrame, train: pd.DataFrame, markers: list[str], stratified: bool = True) -> pd.DataFrame:
    """Quantile-map each marker of `new` onto `train`, optionally within sex x age-band strata."""
    out = new.copy()
    if not stratified:
        for m in markers:
            out[m] = quantile_map(new[m], train[m])
        return out
    s_new, s_tr = strata(new), strata(train)
    for m in markers:
        col = pd.Series(np.nan, index=new.index)
        for s in s_new.unique():
            idx = s_new == s
            col[idx] = quantile_map(new.loc[idx, m], train.loc[s_tr == s, m])
        out[m] = col
    return out


def median_scale(new: pd.DataFrame, train: pd.DataFrame, markers: list[str]) -> pd.DataFrame:
    """Multiply each marker by train median / new median."""
    out = new.copy()
    for m in markers:
        out[m] = new[m] * (train[m].median() / new[m].median())
    return out
