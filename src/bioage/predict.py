"""Biological age for one person from a lab report (SI units). Pure Python; the FastAPI route wraps this.

For personal / educational use. Population-level model; not a diagnosis.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import phenoage as pa

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "xgboost_C.joblib"
MMOL_TO_MGDL_CHOL = 38.67


class LabReport(BaseModel):
    # required: the 9 PhenoAge markers + age (SI units as on most European lab reports)
    age: float = Field(..., ge=20, le=90, description="years")
    albumin_g_l: float = Field(..., gt=10, lt=70)
    creatinine_umol_l: float = Field(..., gt=10, lt=1500)
    glucose_mmol_l: float = Field(..., gt=1, lt=40)
    crp_mg_l: float = Field(..., ge=0, lt=500, description="mg/L (divide by 10 for mg/dL)")
    lymphocyte_pct: float = Field(..., gt=0, lt=100)
    mcv_fl: float = Field(..., gt=50, lt=130)
    rdw_pct: float = Field(..., gt=8, lt=30)
    alp_u_l: float = Field(..., gt=5, lt=1500)
    wbc_10e9_l: float = Field(..., gt=0.5, lt=100, description="10^9/L == 1000 cells/uL")
    # optional extras used by the ML model; missing ones are filled with training medians
    sex: Optional[Literal["female", "male"]] = None
    sbp_mmhg: Optional[float] = Field(None, gt=60, lt=260)
    dbp_mmhg: Optional[float] = Field(None, gt=30, lt=160)
    on_bp_meds: Optional[bool] = None
    total_chol_mmol_l: Optional[float] = Field(None, gt=1, lt=20)
    hdl_mmol_l: Optional[float] = Field(None, gt=0.2, lt=5)
    hba1c_mmol_mol: Optional[float] = Field(None, gt=15, lt=150, description="IFCC units")
    bmi: Optional[float] = Field(None, gt=12, lt=80)
    waist_cm: Optional[float] = Field(None, gt=40, lt=200)
    smoking: Optional[Literal["never", "former", "current"]] = None


class BioAgeResult(BaseModel):
    age: float
    phenoage: float
    phenoage_acceleration: float
    phenoage_10y_aging_death_risk: float
    model_bioage: float
    model_acceleration: float
    model_10y_death_risk: float
    imputed_fields: list[str]
    note: str = ("Population-level estimate from NHANES (US, 1999-2018). Educational use only; "
                 "not a diagnosis. Acceleration is relative to the average American of the same age. "
                 "RDW depends on the lab analyzer: NHANES values rose ~0.9 points after a 2013 analyzer change, "
                 "which alone adds ~3 years to PhenoAge, so treat RDW-driven differences with caution.")


def to_model_row(r: LabReport) -> tuple[pd.DataFrame, list[str]]:
    row = {
        "age": r.age, "albumin": r.albumin_g_l, "creatinine": r.creatinine_umol_l, "glucose": r.glucose_mmol_l,
        "crp": r.crp_mg_l / 10, "lymph_pct": r.lymphocyte_pct, "mcv": r.mcv_fl, "rdw": r.rdw_pct,
        "alp": r.alp_u_l, "wbc": r.wbc_10e9_l,
        "sbp": r.sbp_mmhg,
        "pulse_pressure": (r.sbp_mmhg - r.dbp_mmhg) if r.sbp_mmhg is not None and r.dbp_mmhg is not None else None,
        "on_bp_meds": None if r.on_bp_meds is None else float(r.on_bp_meds),
        "total_chol": None if r.total_chol_mmol_l is None else r.total_chol_mmol_l * MMOL_TO_MGDL_CHOL,
        "hdl": None if r.hdl_mmol_l is None else r.hdl_mmol_l * MMOL_TO_MGDL_CHOL,
        "hba1c": None if r.hba1c_mmol_mol is None else 0.0915 * r.hba1c_mmol_mol + 2.15,   # IFCC -> NGSP %
        "bmi": r.bmi, "waist": r.waist_cm,
        "current_smoker": None if r.smoking is None else float(r.smoking == "current"),
        "ever_smoker": None if r.smoking is None else float(r.smoking != "never"),
        "female": None if r.sex is None else float(r.sex == "female"),
    }
    imputed = [k for k, v in row.items() if v is None]
    return pd.DataFrame([{k: (np.nan if v is None else v) for k, v in row.items()}]), imputed


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)):
    return joblib.load(path)


def predict(report: LabReport) -> BioAgeResult:
    df, imputed = to_model_row(report)
    bundle = load_model()
    model, ref = bundle["model"], bundle["age_ref"]
    risk10 = float(1 - model.surv(df, [10.0])[0, 0])
    bio = float(ref.bio_age(np.array([risk10]))[0])
    pheno = float(pa.phenoage(df).iloc[0])
    return BioAgeResult(
        age=report.age, phenoage=round(pheno, 1), phenoage_acceleration=round(pheno - report.age, 1),
        phenoage_10y_aging_death_risk=round(float(pa.mortality_score(df).iloc[0]), 4),
        model_bioage=round(bio, 1), model_acceleration=round(bio - report.age, 1),
        model_10y_death_risk=round(risk10, 4), imputed_fields=imputed)
