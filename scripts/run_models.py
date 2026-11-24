"""Train every model on every feature set, pick settings on validation, report on test (once).

Usage:  PYTHONPATH=src python scripts/run_models.py
Writes: reports/model_results.csv, reports/figures/*.png, data/processed/predictions.parquet
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see bioage/__init__.py

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter  # noqa: E402
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc  # noqa: E402

from bioage import models as M  # noqa: E402
from bioage.evaluate import bootstrap, calibration_table, point_metrics  # noqa: E402
from bioage.phenoage import phenoage  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GRID = np.arange(1.0, 9.01, 0.25)      # test cycle (2009-10) has 9-11 years follow-up -> 9y horizon
AUC_T = [5, 9]
EXT_AUC_T = [2, 3]
N_BOOT = 200


def load():
    df = pd.read_parquet(ROOT / "data" / "processed" / "nhanes.parquet")
    df = df.dropna(subset=M.PHENO + ["age", "dead", "time_years"])
    df["time_years"] = df["time_years"].clip(lower=1 / 24)
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ["train", "val", "test", "external"]}


def accel_hr(bio, age, time, event):
    """HR for +5 years of age acceleration (bio-age residual on age), adjusted for age."""
    resid = bio - np.poly1d(np.polyfit(age, bio, 1))(age)
    d = pd.DataFrame({"T": time, "E": event, "age": age, "accel5": resid / 5})
    s = CoxPHFitter().fit(d, "T", "E").summary.loc["accel5"]
    return s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"], resid.std()


def main():
    D = load()
    tr, va, te, ex = D["train"], D["val"], D["test"], D["external"]
    print({k: (len(v), int(v.dead.sum())) for k, v in D.items()})
    y_tr, y_te, y_ex = M.y_of(tr), M.y_of(te), M.y_of(ex)
    ref = M.AgeReference().fit(tr)

    runs = [("AgeOnly", "-", M.Cox(["age"])), ("PhenoAge", "A", M.PhenoAgeModel())]
    for fs, feats in M.FEATURE_SETS.items():
        for cls in M.MODELS[1:]:
            runs.append((cls.name, fs, cls(feats)))

    rows, preds, calib, fitted = [], {}, {}, {}
    for name, fs, model in runs:
        print(f"-> {name} [{fs}]", flush=True)
        model.fit(tr, va)
        fitted[(name, fs)] = model
        r_va = model.risk(va)
        val_c = concordance_index_censored(M.y_of(va)["event"], M.y_of(va)["time"], r_va)[0]

        r_te, S_te = model.risk(te), model.surv(te, GRID)
        pm = point_metrics(y_tr, y_te, r_te, S_te, GRID, AUC_T)
        lo, hi = bootstrap(y_tr, y_te, r_te, S_te, GRID, AUC_T, n_boot=N_BOOT)

        # biological age: model's 10-year risk -> age with same risk under age-only reference
        bio_te = phenoage(te).to_numpy() if name == "PhenoAge" else ref.bio_age(1 - model.surv(te, [10.0])[:, 0])
        if name == "AgeOnly":   # bio-age is a function of age alone -> acceleration is identically 0
            hr = hr_lo = hr_hi = sd_acc = np.nan
        else:
            hr, hr_lo, hr_hi, sd_acc = accel_hr(bio_te, te["age"].to_numpy(), te["time_years"].to_numpy(), te["dead"].to_numpy())

        r_ex = model.risk(ex)
        ex_c = concordance_index_censored(y_ex["event"], y_ex["time"], r_ex)[0]
        ex_auc, _ = cumulative_dynamic_auc(y_tr, y_ex, r_ex, EXT_AUC_T)

        row = {"model": name, "features": fs, "val_c": val_c, **pm,
               **{f"{k}_lo": lo[k] for k in pm}, **{f"{k}_hi": hi[k] for k in pm},
               "accel_hr_per5y": hr, "accel_hr_lo": hr_lo, "accel_hr_hi": hr_hi, "accel_sd": sd_acc,
               "corr_bio_age": np.corrcoef(bio_te, te["age"])[0, 1],
               "ext_harrell_c": ex_c, "ext_auc_2y": ex_auc[0], "ext_auc_3y": ex_auc[1]}
        rows.append(row)
        print(f"   val C {val_c:.3f} | test C {pm['harrell_c']:.3f} [{lo['harrell_c']:.3f}-{hi['harrell_c']:.3f}] "
              f"| IBS {pm['ibs']:.4f} | accel HR/5y {hr:.2f} | ext C {ex_c:.3f}", flush=True)

        key = f"{name}_{fs}"
        for split_name, d in [("val", va), ("test", te), ("external", ex)]:
            bio = phenoage(d).to_numpy() if name == "PhenoAge" else ref.bio_age(1 - model.surv(d, [10.0])[:, 0])
            preds.setdefault(split_name, pd.DataFrame({"SEQN": d.SEQN, "cycle": d.cycle}))
            preds[split_name][f"risk_{key}"] = model.risk(d)
            preds[split_name][f"bioage_{key}"] = bio
        calib[key] = calibration_table(S_te[:, -1], te["time_years"].to_numpy(), te["dead"].to_numpy(), GRID[-1])

    res = pd.DataFrame(rows)
    res.to_csv(REP / "model_results.csv", index=False)
    pd.concat([p.assign(split=s) for s, p in preds.items()]).to_parquet(ROOT / "data/processed/predictions.parquet", index=False)

    plot_comparison(res)
    plot_calibration(calib)
    explain(fitted, tr, te)
    save_serving_model(fitted, ref, res)
    print("\n", res[["model", "features", "val_c", "harrell_c", "uno_c", "auc_5y", "auc_9y", "ibs",
                     "accel_hr_per5y", "ext_harrell_c"]].round(3).to_string(index=False))


def plot_comparison(res):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"-": "#888888", "A": "#4C78A8", "B": "#E45756", "C": "#54A24B"}
    labels = res["model"] + "  [" + res["features"] + "]"
    y = np.arange(len(res))[::-1]
    for yi, (_, r) in zip(y, res.iterrows()):
        ax.errorbar(r.harrell_c, yi, xerr=[[r.harrell_c - r.harrell_c_lo], [r.harrell_c_hi - r.harrell_c]],
                    fmt="o", color=colors[r.features], capsize=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Harrell's C-index on test (2009-10), 95% bootstrap CI")
    ax.set_title("Model comparison  (A = PhenoAge markers, B = +BP, C = +lipids/BMI/smoking)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_calibration(calib):
    keys = [k for k in calib if k.endswith("_B") or k.startswith(("PhenoAge", "AgeOnly"))]
    n = len(keys)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8), sharex=True, sharey=True)
    for ax, k in zip(axes.flat, keys):
        c = calib[k]
        ax.plot([0, 0.7], [0, 0.7], "k--", lw=1)
        ax.plot(c.predicted, c.observed, "o-")
        ax.set_title(k.replace("_", " [") + "]", fontsize=10)
        ax.set_xlabel("predicted 9-yr risk")
        ax.set_ylabel("observed (Kaplan-Meier)")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Calibration on test set, 9-year horizon (feature set B)")
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=150)
    plt.close(fig)


def save_serving_model(fitted, ref, res):
    """Persist XGBoost-C + the age reference for bioage.predict / the FastAPI route."""
    import joblib
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    row = res[(res.model == "XGBoost") & (res.features == "C")].iloc[0]
    joblib.dump({"model": fitted[("XGBoost", "C")], "age_ref": ref, "features": M.FEATURE_SETS["C"],
                 "test_c_index": float(row.harrell_c), "trained_on": "NHANES 1999-2006, early-stopped on 2007-08"},
                out / "xgboost_C.joblib")


def explain(fitted, tr, te):
    import shap
    # Cox hazard ratios (set C), per 1 SD
    cox = fitted[("Cox", "C")]
    hr = pd.Series(np.exp(cox.m.coef_), index=cox.prep.features).sort_values()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(hr.index, hr.values - 1, left=1, color=np.where(hr > 1, "#E45756", "#4C78A8"))
    ax.axvline(1, color="k", lw=1)
    ax.set_xlabel("hazard ratio per 1 SD (adjusted for all other features)")
    ax.set_title("Cox model, feature set C")
    fig.tight_layout()
    fig.savefig(FIG / "cox_hazard_ratios.png", dpi=150)
    plt.close(fig)
    hr.rename("hr_per_sd").to_csv(REP / "cox_hazard_ratios.csv")

    # XGBoost SHAP (set C)
    xm = fitted[("XGBoost", "C")]
    Xte = pd.DataFrame(xm.prep.transform(te), columns=xm.prep.features)
    sample = Xte.sample(min(2000, len(Xte)), random_state=0)
    sv = shap.TreeExplainer(xm.m).shap_values(sample)
    plt.figure()
    shap.summary_plot(sv, sample, show=False, max_display=20)
    plt.title("XGBoost (set C): SHAP, log-hazard contribution")
    plt.tight_layout()
    plt.savefig(FIG / "shap_xgboost.png", dpi=150)
    plt.close("all")

    # DeepSurv SHAP (set C)
    import torch
    dm = fitted[("DeepSurv", "C")]
    Xtr_t = torch.tensor(dm.prep.transform(tr.sample(500, random_state=0)))
    Xs = torch.tensor(dm.prep.transform(te.sample(1000, random_state=0)))
    dm.net.eval()
    sv = shap.GradientExplainer(dm.net, Xtr_t).shap_values(Xs)
    sv = np.asarray(sv).reshape(len(Xs), -1)
    plt.figure()
    shap.summary_plot(sv, pd.DataFrame(Xs.numpy(), columns=dm.prep.features), show=False, max_display=20)
    plt.title("DeepSurv (set C): SHAP")
    plt.tight_layout()
    plt.savefig(FIG / "shap_deepsurv.png", dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
