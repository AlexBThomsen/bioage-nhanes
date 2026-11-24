"""Unit tests for the core math and parsing that every result depends on."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import CoxPHFitter

from bioage import models as M
from bioage.build import derive
from bioage.evaluate import point_metrics

ROOT = Path(__file__).resolve().parents[1]


def cox_data(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 85, n)
    x = rng.normal(size=n)
    hazard = 0.0005 * np.exp(0.08 * (age - 20) + 0.4 * x)
    t_event = rng.exponential(1 / hazard)
    c = rng.uniform(5, 20, n)
    return pd.DataFrame({"age": age, "x": x, "time_years": np.minimum(t_event, c),
                         "dead": (t_event <= c).astype(int)})


# ---------------------------------------------------------------- survival machinery
def test_breslow_matches_lifelines_baseline():
    d = cox_data()
    cph = CoxPHFitter(baseline_estimation_method="breslow").fit(d[["age", "x", "time_years", "dead"]],
                                                               "time_years", "dead")
    risk = d[["age", "x"]].to_numpy() @ cph.params_.to_numpy()
    base = M.Breslow().fit(risk, d.time_years.to_numpy(), d.dead.astype(bool).to_numpy())
    times = np.array([2.0, 5.0, 10.0])
    ours = base.surv(risk[:5], times)
    theirs = cph.predict_survival_function(d[["age", "x"]].iloc[:5], times=times).T.to_numpy()
    assert np.allclose(ours, theirs, atol=2e-3)


def test_age_reference_inverts_itself():
    """A person whose 10-yr risk equals the age-only model's risk at age a must get bio-age a."""
    d = cox_data()
    ref = M.AgeReference().fit(d)
    ages = np.array([30.0, 50.0, 70.0])
    risk10 = 1 - np.exp(-np.exp(ref.logH + ref.b * ages))
    assert np.allclose(ref.bio_age(risk10), ages, atol=1e-6)


def test_bio_age_monotone_in_risk():
    ref = M.AgeReference().fit(cox_data())
    bio = ref.bio_age(np.array([0.01, 0.05, 0.2, 0.6]))
    assert np.all(np.diff(bio) > 0)


def test_prep_fits_on_train_only():
    tr = pd.DataFrame({"crp": [0.1, 0.2, np.nan], "age": [30, 40, 50]})
    te = pd.DataFrame({"crp": [np.nan, 100.0], "age": [60, 70]})
    p = M.Prep(["crp", "age"]).fit(tr)
    # imputation value = train median on the log scale
    assert np.isclose(p.med["crp"], np.mean([np.log(0.1), np.log(0.2)]))
    X = p.transform(te)
    X_other = p.transform(te.assign(crp=[np.nan, 1e6]))          # a wild test outlier must not leak in
    expected = (p.med["crp"] - p.mu["crp"]) / p.sd["crp"]
    assert np.isclose(X[0, 0], expected, atol=1e-5) and np.isclose(X_other[0, 0], expected, atol=1e-5)


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_and_random():
    d = cox_data(3000)
    y = M.y_of(d)
    grid = np.arange(1.0, 9.01, 0.5)
    true_risk = 0.08 * d.age + 0.4 * d.x
    base = M.Breslow().fit(true_risk.to_numpy(), d.time_years.to_numpy(), d.dead.astype(bool).to_numpy())
    S = base.surv(true_risk.to_numpy(), grid)
    good = point_metrics(y, y, true_risk.to_numpy(), S, grid, [5])
    noise = np.random.default_rng(0).normal(size=len(d))
    bad = point_metrics(y, y, noise, np.tile(S.mean(0), (len(d), 1)), grid, [5])
    assert good["harrell_c"] > 0.75 and abs(bad["harrell_c"] - 0.5) < 0.03
    assert good["ibs"] < bad["ibs"]


# ---------------------------------------------------------------- derived variables
def test_derive_bp_stage_and_smoking():
    raw = pd.DataFrame({
        "cycle": [2009] * 5, "sex": [1, 2, 1, 2, 1],
        "sbp": [118, 125, 135, 150, np.nan], "dbp": [70, 75, 70, 80, np.nan],
        "smq020": [2, 1, 1, 1, 9], "smq040": [np.nan, 1, 3, 2, np.nan],
        "bpq020": [2, 1, 1, 1, 9], "bpq050a": [np.nan, 1, 2, np.nan, np.nan],
        "crp_mgdl": [0.1] * 5, "hscrp_mgl": [np.nan] * 5, "creatinine": [80.0] * 5,
        "diq010": [2] * 5, "mcq_stroke": [2] * 5, "mcq_chd": [2] * 5, "mcq_mi": [2] * 5,
        "mcq_chf": [2] * 5, "mcq_cancer": [2] * 5,
        "mortstat": [0, 1, 0, 1, 0], "permth_exm": [100, 50, 100, 20, 100], "ucod_leading": [np.nan, 5, np.nan, 1, np.nan],
        "wt2": [1.0] * 5, "wt4": [np.nan] * 5, "age": [50] * 5,
    })
    d = derive(raw)
    assert d.bp_stage.tolist()[:4] == ["normal", "elevated", "stage1", "stage2"]
    assert d.bp_stage.iloc[4] is None
    assert d.smoking.tolist()[:4] == ["never", "current", "former", "current"]
    assert np.isnan(d.current_smoker.iloc[4])                    # refused/unknown stays missing
    assert d.on_bp_meds.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert d.stroke_death.tolist()[:4] == [0.0, 1.0, 0.0, 0.0]
    assert np.isclose(d.pulse_pressure.iloc[3], 70)


def test_creatinine_recalibration_only_affects_1999_and_2005():
    base = dict(sex=1, sbp=120, dbp=80, smq020=2, smq040=np.nan, bpq020=2, bpq050a=np.nan, crp_mgdl=0.1,
                hscrp_mgl=np.nan, diq010=2, mcq_stroke=2, mcq_chd=2, mcq_mi=2, mcq_chf=2, mcq_cancer=2,
                mortstat=0, permth_exm=100, ucod_leading=np.nan, wt2=1.0, wt4=1.0, age=50, creatinine=0.8 * 88.4017)
    raw = pd.DataFrame([dict(base, cycle=c) for c in (1999, 2001, 2005, 2009)])
    cr = derive(raw).creatinine.to_numpy() / 88.4017
    assert np.allclose(cr, [1.013 * 0.8 + 0.147, 0.8, -0.016 + 0.978 * 0.8, 0.8])


# ---------------------------------------------------------------- NHANES III parser
def test_nhanes3_missing_code_rule(tmp_path, monkeypatch):
    from bioage import nhanes3
    sas = tmp_path / "lab.sas"
    sas.write_text("DATA X;\n     INPUT\n        SEQN     1-5\n        CRP      6-10\n        WAIST    11-15\n;\n")
    dat = tmp_path / "lab.dat"
    dat.write_text("00001 0.21 88.8\n0000288888888.8\n00003 8.88 99.9\n")
    monkeypatch.setattr(nhanes3, "D3", tmp_path)
    monkeypatch.setitem(nhanes3.VARS, "lab", ["SEQN", "CRP", "WAIST"])
    out = nhanes3.read_fixed("lab")
    assert out.CRP.iloc[0] == 0.21 and np.isnan(out.CRP.iloc[1]) and out.CRP.iloc[2] == 8.88
    assert out.WAIST.iloc[0] == 88.8                            # real value with leading space is kept
    assert np.isnan(out.WAIST.iloc[1])                           # '888.8' filling the field = missing


# ---------------------------------------------------------------- API unit conversions
def test_lab_report_unit_conversions():
    from bioage.predict import LabReport, to_model_row
    r = LabReport(age=50, albumin_g_l=45, creatinine_umol_l=80, glucose_mmol_l=5, crp_mg_l=2.0,
                  lymphocyte_pct=30, mcv_fl=90, rdw_pct=13, alp_u_l=70, wbc_10e9_l=6,
                  total_chol_mmol_l=5.0, hba1c_mmol_mol=42, sbp_mmhg=130, dbp_mmhg=80, smoking="former")
    row, imputed = to_model_row(r)
    assert np.isclose(row.crp.iloc[0], 0.2)                      # mg/L -> mg/dL
    assert np.isclose(row.total_chol.iloc[0], 193.35)            # mmol/L -> mg/dL
    assert np.isclose(row.hba1c.iloc[0], 0.0915 * 42 + 2.15)     # IFCC -> NGSP (~6.0%)
    assert row.pulse_pressure.iloc[0] == 50
    assert row.current_smoker.iloc[0] == 0 and row.ever_smoker.iloc[0] == 1
    assert set(imputed) == {"on_bp_meds", "hdl", "bmi", "waist", "female"}


# ---------------------------------------------------------------- golden numbers (catch silent drift)
@pytest.mark.skipif(not (ROOT / "data/processed/nhanes.parquet").exists(), reason="run bioage.build first")
def test_golden_phenoage_reproduction():
    """Headline reproduction must stay where it was when the report was written (Nov 2026)."""
    from bioage.phenoage import MARKERS, phenoage
    df = pd.read_parquet(ROOT / "data/processed/nhanes.parquet")
    d = df[df.cycle <= 2009].dropna(subset=MARKERS + ["age", "dead", "time_years"]).copy()
    assert 28_000 < len(d) < 29_000
    d["pa"] = phenoage(d)
    d["accel"] = d.pa - np.poly1d(np.polyfit(d.age, d.pa, 1))(d.age)
    hr = np.exp(CoxPHFitter().fit(d[["time_years", "dead", "age", "accel"]], "time_years", "dead").params_["accel"])
    assert 1.040 < hr < 1.055                                    # literature ~1.045
    assert 0.91 < d[["pa", "age"]].corr().iloc[0, 1] < 0.94


@pytest.mark.skipif(not (ROOT / "reports/model_results.csv").exists(), reason="run run_models.py first")
def test_golden_model_results():
    r = pd.read_csv(ROOT / "reports/model_results.csv").set_index(["model", "features"])
    assert abs(r.loc[("XGBoost", "C"), "harrell_c"] - 0.873) < 0.005
    assert abs(r.loc[("PhenoAge", "A"), "harrell_c"] - 0.858) < 0.005
    assert r.loc[("AgeOnly", "-"), "harrell_c"] < r.loc[("PhenoAge", "A"), "harrell_c"]
