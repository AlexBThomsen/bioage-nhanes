"""Analysis 1 (pre-registered, PREREGISTRATION.md): does the full NHANES lab panel improve mortality prediction?

Leave-one-era-out over five NHANES eras, follow-up censored at 10 years, XGBoost-Cox with fixed settings.
Usage:  PYTHONPATH=src python scripts/full_panel.py
Writes: reports/full_panel_results.json, reports/figures/full_panel_*.png
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from assay_drift import harmonize_rdw  # noqa: E402
from bioage import models as M  # noqa: E402
from bioage.phenoage import MARKERS, linear_predictor  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
ERAS = {1999: "1999-02", 2001: "1999-02", 2003: "2003-06", 2005: "2003-06", 2007: "2007-10", 2009: "2007-10",
        2011: "2011-14", 2013: "2011-14", 2015: "2015-18", 2017: "2015-18"}
HORIZON = 10.0
N_BOOT = 500
PARAMS = dict(objective="survival:cox", eval_metric="cox-nloglik", eta=0.03, max_depth=3, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, seed=0, nthread=1)
EXTRAS = ["sbp", "pulse_pressure", "on_bp_meds", "total_chol", "hdl", "hba1c", "bmi", "waist",
          "current_smoker", "ever_smoker", "female"]


def load():
    df = pd.read_parquet(ROOT / "data/processed/nhanes.parquet")
    df = harmonize_rdw(df)                                    # pre-registered: RDW 2013+ -> 2011-12 distribution
    req = [m for m in MARKERS if m != "crp"] + ["age", "dead", "time_years"]
    df = df.dropna(subset=req).copy()
    df["era"] = df["cycle"].map(ERAS)
    df["dead10"] = ((df["dead"] == 1) & (df["time_years"] <= HORIZON)).astype(int)
    df["t10"] = df["time_years"].clip(lower=1 / 24, upper=HORIZON)
    return df.reset_index(drop=True)


def feature_sets(df):
    lab = sorted(c for c in df if c.startswith("lab_"))
    a = MARKERS + ["age"]
    return {"M2": a, "M3": a + lab, "M4": a + lab + EXTRAS}


def fit_xgb(tr, feats, seed=0):
    rng = np.random.default_rng(seed)
    inner = rng.random(len(tr)) < 0.15
    def dm(d):
        X = d[feats].to_numpy(float)
        X[:, feats.index("crp")] = np.log(np.clip(X[:, feats.index("crp")], 0.01, None))
        y = np.where(d["dead10"] == 1, d["t10"], -d["t10"])
        return xgb.DMatrix(X, label=y, missing=np.nan)
    dtr, dva = dm(tr[~inner]), dm(tr[inner])
    booster = xgb.train(PARAMS, dtr, num_boost_round=3000, evals=[(dva, "val")],
                        early_stopping_rounds=100, verbose_eval=False)
    return booster, dm


def cidx(d, risk):
    return concordance_index_censored(d["dead10"].astype(bool).to_numpy(), d["t10"].to_numpy(), risk)[0]


def plot_results(out):
    per_era, pooled = out["eras"], out["pooled"]
    eras = sorted(per_era)
    imp = pd.Series(out["shap_mean_abs_top15"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(eras))
    for off, key, col, lab in [(0.15, "M3-M2", "#4C78A8", "full panel − 9 markers"),
                               (-0.15, "M4-M3", "#54A24B", "+ BP & lifestyle − full panel")]:
        v = np.array([per_era[e]["delta"].get(key, [np.nan] * 3) for e in eras])
        ax.errorbar(v[:, 0], y + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=col, capsize=3, label=lab)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(0.005, color="grey", ls=":", lw=1)
    ax.set_yticks(y, [f"held out {e}\n(n={per_era[e]['n']:,}, {per_era[e]['deaths_10y']} deaths)" for e in eras])
    ax.set_xlabel("Δ C-index in the held-out era (paired bootstrap 95% CI)")
    ax.set_title(f"Leave-one-era-out. Pooled full panel − 9 markers: {pooled['M3-M2'][0]:+.4f} "
                 f"({pooled['M3-M2'][1]:+.4f} to {pooled['M3-M2'][2]:+.4f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    top = imp.head(15)[::-1]
    new = [c.startswith("lab_") for c in top.index]
    ax2.barh([c.replace("lab_", "") for c in top.index], top.values, color=["#E45756" if n else "#9aa6b1" for n in new])
    ax2.set_xlabel("mean |SHAP| (log-hazard)")
    ax2.set_title("Most important markers (red = new beyond PhenoAge's nine)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "full_panel.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    fs = feature_sets(df)
    print(f"n={len(df)}, deaths within 10y={df.dead10.sum()}, extra markers={len(fs['M3']) - len(fs['M2'])}", flush=True)
    rng = np.random.default_rng(0)
    eras = sorted(df.era.unique())
    per_era, boots = {}, {}
    for era in eras:
        tr, te = df[df.era != era], df[df.era == era].reset_index(drop=True)
        tr10 = tr.assign(dead=tr.dead10, time_years=tr.t10)
        risks = {"M0": M.Cox(["age"]).fit(tr10, tr10).risk(te)}
        m1 = linear_predictor(te).to_numpy()                 # NaN where CRP is missing
        risks["M1"] = m1 if np.isfinite(m1).mean() > 0.5 else None   # 2011-14: no CRP at all
        iters = {}
        for name, feats in fs.items():
            booster, dm = fit_xgb(tr, feats)
            risks[name] = booster.predict(dm(te), output_margin=True, iteration_range=(0, booster.best_iteration + 1))
            iters[name] = booster.best_iteration
        e, t = te["dead10"].astype(bool).to_numpy(), te["t10"].to_numpy()
        def c_on(mask, r):
            return concordance_index_censored(e[mask], t[mask], r[mask])[0]
        cs = {k: (float(c_on(np.isfinite(r), r)) if r is not None else None) for k, r in risks.items()}
        pairs = [("M3", "M2"), ("M2", "M1"), ("M4", "M3"), ("M2", "M0")]
        valid = [(a, b) for a, b in pairs if risks[a] is not None and risks[b] is not None]
        masks = {(a, b): np.isfinite(risks[a]) & np.isfinite(risks[b]) for a, b in valid}   # pairwise-complete
        point = {f"{a}-{b}": float(c_on(masks[(a, b)], risks[a]) - c_on(masks[(a, b)], risks[b])) for a, b in valid}
        deltas = {f"{a}-{b}": [] for a, b in valid}
        for _ in range(N_BOOT):
            i = rng.integers(0, len(te), len(te))
            for a, b in valid:
                mi = i[masks[(a, b)][i]]
                deltas[f"{a}-{b}"].append(concordance_index_censored(e[mi], t[mi], risks[a][mi])[0]
                                          - concordance_index_censored(e[mi], t[mi], risks[b][mi])[0])
        boots[era] = {k: np.array(v) for k, v in deltas.items()}
        per_era[era] = {"n": int(len(te)), "deaths_10y": int(te.dead10.sum()), "c": cs, "best_iter": iters,
                        "n_with_crp": int(np.isfinite(m1).sum()),
                        "delta": {k: [point[k], *map(float, np.percentile(v, [2.5, 97.5]))] for k, v in boots[era].items()}}
        print(era, {k: round(v, 3) if v else None for k, v in cs.items()},
              {k: [round(x, 4) for x in v] for k, v in per_era[era]["delta"].items()}, flush=True)

    pooled = {}
    for key in ["M3-M2", "M4-M3", "M2-M0", "M2-M1"]:
        es = [e for e in eras if key in boots[e]]
        w = np.array([per_era[e]["n"] for e in es], float)
        w /= w.sum()
        point = float(sum(wi * per_era[e]["delta"][key][0] for wi, e in zip(w, es)))
        rep = sum(wi * boots[e][key] for wi, e in zip(w, es))
        pooled[key] = [point, *map(float, np.percentile(rep, [2.5, 97.5])), es]
    h8 = pooled["M3-M2"][0] >= 0.005 and pooled["M3-M2"][1] > 0
    h9_eras = {e: per_era[e]["delta"]["M2-M1"] for e in eras if "M2-M1" in per_era[e]["delta"]}
    h9 = all(v[1] > 0 for v in h9_eras.values())
    out = {"n": int(len(df)), "eras": per_era, "pooled": pooled,
           "H8_supported": bool(h8), "H9_supported_in_evaluable_eras": bool(h9),
           "H9_note": "PhenoAge needs CRP, which was not measured in 2011-14, so H9 is evaluable in 4 of 5 eras"}

    # which new markers matter: SHAP on M3 trained on all eras
    import shap
    booster, dm = fit_xgb(df, fs["M3"])
    sample = df.sample(3000, random_state=0)
    sv = shap.TreeExplainer(booster).shap_values(dm(sample))
    imp = pd.Series(np.abs(sv).mean(0), index=fs["M3"]).sort_values(ascending=False)
    out["shap_mean_abs_top15"] = imp.head(15).round(4).to_dict()
    (REP / "full_panel_results.json").write_text(json.dumps(out, indent=2, default=float))

    plot_results(out)
    print(json.dumps({"pooled": pooled, "H8": h8, "H9": h9, "top": out["shap_mean_abs_top15"]}, indent=1, default=str))


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        plot_results(json.loads((REP / "full_panel_results.json").read_text()))
    else:
        main()
