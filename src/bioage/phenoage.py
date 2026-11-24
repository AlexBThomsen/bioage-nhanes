"""Levine et al. 2018 Phenotypic Age (blood-chemistry version).

Note: the formula was trained on *aging-related* mortality. Its predicted risk matches observed 10-year
mortality excluding accidents (004), flu/pneumonia (008) and other causes (010): 0.110 vs 0.109 in
NHANES 1999-2006, while all-cause mortality is 0.150. So it under-predicts all-cause risk by design.

Inputs (units matter!): albumin g/L, creatinine umol/L, glucose mmol/L, CRP mg/dL (log-transformed here),
lymphocyte %, MCV fL, RDW %, alkaline phosphatase U/L, WBC 1000 cells/uL, age years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COEF = {   # exact values from the BioAge R package (Kwon & Belsky 2021), phenoage_calc(orig=TRUE)
    "albumin": -0.03359355,
    "creatinine": 0.009506491,
    "glucose": 0.1953192,
    "log_crp": 0.09536762,
    "lymph_pct": -0.01199984,
    "mcv": 0.02676401,
    "rdw": 0.3306156,
    "alp": 0.001868778,
    "wbc": 0.05542406,
    "age": 0.08035356,
}
INTERCEPT = -19.90667
GAMMA = 0.007692696
MARKERS = ["albumin", "creatinine", "glucose", "crp", "lymph_pct", "mcv", "rdw", "alp", "wbc"]


def linear_predictor(df: pd.DataFrame) -> pd.Series:
    x = df.assign(log_crp=np.log(df["crp"].clip(lower=0.01)))   # floor avoids log(0) for below-LOD values
    return INTERCEPT + sum(c * x[k] for k, c in COEF.items())


def mortality_score(df: pd.DataFrame, months: float = 120) -> pd.Series:
    """Gompertz-model probability of death within `months`."""
    xb = linear_predictor(df)
    return 1 - np.exp(-np.exp(xb) * (np.exp(GAMMA * months) - 1) / GAMMA)


def phenoage(df: pd.DataFrame) -> pd.Series:
    m = mortality_score(df).clip(upper=1 - 1e-12)
    # NB: the denominator is 0.090165 (an earlier version of this file had 0.09165, ~1.5 yr too young)
    return 141.50225 + np.log(-0.0055305 * np.log(1 - m)) / 0.090165
