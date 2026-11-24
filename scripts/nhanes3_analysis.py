"""NHANES III (1988-94): PhenoAge on its own training cohort, backward validation, pooled stroke analysis.

Usage:  PYTHONPATH=src python -m bioage.nhanes3 && PYTHONPATH=src python scripts/nhanes3_analysis.py
        (needs models/xgboost_C.joblib from run_models.py)
Writes: reports/nhanes3_results.json, reports/figures/nhanes3_*.png
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see bioage/__init__.py

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import bp_deepdive as B  # noqa: E402
from bioage import models as M  # noqa: E402
from bioage.phenoage import MARKERS, mortality_score, phenoage  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
OUT: dict = {}
AGING_EXCLUDED = [4, 8, 10]   # accidents, flu/pneumonia, other causes


def km_risk(t, e, h):
    return 1 - float(KaplanMeierFitter().fit(t, e).survival_function_at_times(h).iloc[0])


def phenoage_check(n3):
    d = n3.dropna(subset=MARKERS + ["age", "dead"]).copy()
    d["pa"] = phenoage(d)
    d["M"] = mortality_score(d)
    d["accel"] = d.pa - np.poly1d(np.polyfit(d.age, d.pa, 1))(d.age)
    cph = CoxPHFitter().fit(d[["time_years", "dead", "age", "accel"]], "time_years", "dead").summary.loc["accel"]
    aging = ((d.dead == 1) & ~d.cause.isin(AGING_EXCLUDED)).astype(int)
    res = {"n": int(len(d)), "corr_with_age": float(d[["pa", "age"]].corr().iloc[0, 1]),
           "mean_phenoage_minus_age": float((d.pa - d.age).mean()),
           "hr_per_year_accel": [float(cph["exp(coef)"]), float(cph["exp(coef) lower 95%"]), float(cph["exp(coef) upper 95%"])],
           "mean_predicted_10y_M": float(d.M.mean()),
           "observed_10y_all_cause": km_risk(d.time_years, d.dead, 10),
           "observed_10y_aging_causes": km_risk(d.time_years, aging, 10)}
    q = pd.qcut(d.M, 10, labels=False)
    cal = pd.DataFrame([{"pred": d.M[q == g].mean(),
                         "obs_all": km_risk(d.time_years[q == g], d.dead[q == g], 10),
                         "obs_aging": km_risk(d.time_years[q == g], aging[q == g], 10)} for g in range(10)])
    res["calibration_deciles"] = cal.round(4).to_dict(orient="list")
    OUT["1_phenoage_on_training_cohort"] = res

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 0.8], [0, 0.8], "k--", lw=1)
    ax.plot(cal.pred, cal.obs_all, "o-", label="all-cause death")
    ax.plot(cal.pred, cal.obs_aging, "s-", label="excl. accidents, flu/pneumonia, other")
    ax.set_xlabel("PhenoAge predicted 10-yr risk (M)")
    ax.set_ylabel("observed 10-yr risk (Kaplan–Meier)")
    ax.set_title(f"PhenoAge calibration, NHANES III (its training cohort)\nn={len(d):,}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nhanes3_phenoage_calibration.png", dpi=150)
    plt.close(fig)


def backward_validation(n3, main):
    """Models trained on 1999-2006 scored on 1988-94 participants."""
    bundle = joblib.load(ROOT / "models" / "xgboost_C.joblib")
    xgb_c, ref = bundle["model"], bundle["age_ref"]
    d = n3.dropna(subset=MARKERS + ["age", "dead", "time_years"]).copy()
    d = d[d.age <= 85]                        # 1999-2006 training data is top-coded at 85
    d["time_years"] = d["time_years"].clip(lower=1 / 24)
    y_full = M.y_of(d)                         # untruncated follow-up: used only for the IPCW censoring curve
    # administrative censoring at 20 y so follow-up matches the training data's range
    d["dead"] = ((d["dead"] == 1) & (d["time_years"] <= 20)).astype(int)
    d["time_years"] = d["time_years"].clip(upper=20)
    y = M.y_of(d)
    tr = main[(main.split == "train")].dropna(subset=MARKERS + ["age", "dead", "time_years"])
    y_tr = y_full                              # censoring distribution estimated in NHANES III itself
    age_only = M.Cox(["age"]).fit(tr, tr)
    rows = {}
    for name, risk, s10 in [
        ("age only", age_only.risk(d), age_only.surv(d, [10.0])[:, 0]),
        ("PhenoAge (published)", M.PhenoAgeModel().risk(d), M.PhenoAgeModel().surv(d, [10.0])[:, 0]),
        ("XGBoost-C (ours)", xgb_c.risk(d), xgb_c.surv(d, [10.0])[:, 0]),
    ]:
        c20 = concordance_index_censored(y["event"], y["time"], risk)[0]
        auc10, _ = cumulative_dynamic_auc(y_tr, y, 1 - s10, [10.0])
        rows[name] = {"harrell_c_20y": float(c20), "auc_10y": float(auc10[0]),
                      "mean_pred_10y": float((1 - s10).mean())}
    rows["observed_10y_all_cause"] = km_risk(d.time_years, d.dead, 10)
    bio = ref.bio_age(1 - xgb_c.surv(d, [10.0])[:, 0])
    resid = bio - np.poly1d(np.polyfit(d.age, bio, 1))(d.age)
    hr = CoxPHFitter().fit(pd.DataFrame({"T": d.time_years, "E": d.dead, "age": d.age, "a5": resid / 5}),
                           "T", "E").summary.loc["a5"]
    rows["xgb_bioage_accel_hr_per5y"] = [float(hr["exp(coef)"]), float(hr["exp(coef) lower 95%"]), float(hr["exp(coef) upper 95%"])]
    rows["n"] = int(len(d))
    OUT["2_backward_validation"] = rows


def pooled_stroke(n3, main):
    p = B.prep(pd.concat([main[main.cycle <= 2013], n3], ignore_index=True))
    d = p.dropna(subset=["sbp", "dbp", "dead"] + B.FULL_COVS).copy()
    d["cause"] = np.select([d.stroke_death == 1, d.dead == 1], [1, 2], 0)
    d["is_stroke"] = (d.cause == 1).astype(int)
    d["nh3"] = (d.cycle == 1988).astype(float)
    covs = B.FULL_COVS + ["nh3"]               # survey-era indicator
    res = {"n": int(len(d)), "stroke_deaths": int(d.is_stroke.sum()),
           "stroke_deaths_nhanes3": int(d[d.nh3 == 1].is_stroke.sum())}
    hr, _ = B.cox_hr(d, "time_years", "is_stroke", ["sbp10"] + covs, ["sbp10"])
    res["cs_hr_per_10mmHg"] = hr["sbp10"]
    res["fg_shr_per_10mmHg"] = B.fg_hr(d, ["sbp10"] + covs, ["sbp10"])["sbp10"]
    for lab, lo, hi in [("20-59", 20, 59), ("60-74", 60, 74), ("75+", 75, 120)]:
        sub = d[(d.age >= lo) & (d.age <= hi)]
        h, _ = B.cox_hr(sub, "time_years", "is_stroke", ["sbp10"] + covs, ["sbp10"])
        res[f"cs_hr_per_10mmHg_age_{lab}"] = h["sbp10"] + [int(sub.is_stroke.sum())]
    n3only = d[d.nh3 == 1]
    h, _ = B.cox_hr(n3only, "time_years", "is_stroke", ["sbp10"] + B.FULL_COVS, ["sbp10"])
    res["nhanes3_only_cs_hr_per_10mmHg"] = h["sbp10"]
    for s in B.STAGES[1:]:
        d[f"st_{s}"] = (d.bp_stage == s).astype(float)
    st = [f"st_{s}" for s in B.STAGES[1:]]
    h, _ = B.cox_hr(d, "time_years", "is_stroke", st + covs, st)
    res["cs_hr_by_stage_vs_normal"] = h
    pps, _ = B.cox_hr(d, "time_years", "is_stroke", ["sbp10", "pp10"] + covs, ["pp10"])
    ppa, _ = B.cox_hr(d.assign(e=d.dead.astype(int)), "time_years", "e", ["sbp10", "pp10"] + covs, ["pp10"])
    res["stroke_hr_per_10mmHg_pp_given_sbp"] = pps["pp10"]
    res["allcause_hr_per_10mmHg_pp_given_sbp"] = ppa["pp10"]

    a = B.accel_frame(B.prep(n3))
    m = smf.ols(f"accel ~ sbp10 + pp10 + cr(age, df=6) + {B.COVF}", a).fit(cov_type="HC3")
    m2 = smf.ols(f"accel ~ sbp10 + cr(age, df=6) + {B.COVF}", a[a.sbp >= 120]).fit(cov_type="HC3")
    res["nhanes3_bioage_per_10mmHg_pp_given_sbp"] = B.ols_ci(m, "pp10")
    res["nhanes3_bioage_per_10mmHg_sbp_120plus"] = B.ols_ci(m2, "sbp10")
    OUT["3_pooled_stroke_and_pp"] = res

    labels = ["20-59", "60-74", "75+"]
    v = np.array([res[f"cs_hr_per_10mmHg_age_{g}"][:3] for g in labels])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(v[:, 0], range(3), xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color="#E45756", capsize=3)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(range(3), [f"age {g}\n({res[f'cs_hr_per_10mmHg_age_{g}'][3]} stroke deaths)" for g in labels])
    ax.set_xlabel("stroke death HR per +10 mmHg systolic BP")
    ax.set_title(f"NHANES III + 1999–2014 pooled ({res['stroke_deaths']} stroke deaths)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "nhanes3_stroke_by_age.png", dpi=150)
    plt.close(fig)


def main():
    n3 = pd.read_parquet(ROOT / "data" / "processed" / "nhanes3.parquet")
    main_df = pd.read_parquet(ROOT / "data" / "processed" / "nhanes.parquet")
    for step, args in [(phenoage_check, (n3,)), (backward_validation, (n3, main_df)), (pooled_stroke, (n3, main_df))]:
        print(f"-> {step.__name__}", flush=True)
        step(*args)
    (REP / "nhanes3_results.json").write_text(json.dumps(OUT, indent=2, default=float))
    print(json.dumps({k: v for k, v in OUT.items()}, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
