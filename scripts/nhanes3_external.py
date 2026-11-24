"""External validation in NHANES III (pre-registered: PREREGISTRATION.md, H11-H12).

Models are trained once on NHANES 1999-2018 and scored, untouched, on NHANES III (1988-94).
Usage:  PYTHONPATH=src python -m bioage.nhanes3 && PYTHONPATH=src python scripts/nhanes3_external.py
Writes: reports/nhanes3_external.json, reports/figures/nhanes3_external_*.png
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
from lifelines import KaplanMeierFitter  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import full_panel as F  # noqa: E402
from bioage import models as M  # noqa: E402
from bioage.phenoage import MARKERS, linear_predictor  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
N_BOOT = 1000
UNMAPPED = ["lab_neut_pct", "lab_mono_pct", "lab_eos_pct", "lab_baso_pct"]   # differential % measured differently


def load_n3():
    d = pd.read_parquet(ROOT / "data/processed/nhanes3.parquet")
    d = d.dropna(subset=MARKERS + ["age", "dead", "time_years"])
    d = d[d.age <= 85].copy()
    d["dead10"] = ((d.dead == 1) & (d.time_years <= F.HORIZON)).astype(int)
    d["t10"] = d.time_years.clip(lower=1 / 24, upper=F.HORIZON)
    return d.reset_index(drop=True)


def feature_sets(df):
    lab = sorted(c for c in df if c.startswith("lab_") and c not in UNMAPPED)
    a = MARKERS + ["age"]
    return {"M2": a, "M3x": a + lab, "M4x": a + lab + F.EXTRAS}


def fit_all(tr, fs):
    out = {}
    for name, feats in fs.items():
        booster, dm = F.fit_xgb(tr, feats)
        out[name] = (booster, dm)
    return out


def predict(models, te):
    return {k: b.predict(dm(te), output_margin=True, iteration_range=(0, b.best_iteration + 1))
            for k, (b, dm) in models.items()}


def km10(t, e):
    return 1 - float(KaplanMeierFitter().fit(t, e).survival_function_at_times(F.HORIZON).iloc[0])


def main():
    main_df, n3 = F.load(), load_n3()
    fs = feature_sets(main_df)
    print(f"train n={len(main_df)}, NHANES III n={len(n3)}, 10y deaths={n3.dead10.sum()}, "
          f"extra markers={len(fs['M3x']) - len(fs['M2'])}", flush=True)
    models = fit_all(main_df, fs)
    risks = {"M0": n3.age.to_numpy(float), "M1": linear_predictor(n3).to_numpy(), **predict(models, n3)}
    e, t = n3.dead10.astype(bool).to_numpy(), n3.t10.to_numpy()
    c = {k: float(concordance_index_censored(e, t, r)[0]) for k, r in risks.items()}

    pairs = [("M3x", "M2"), ("M2", "M1"), ("M4x", "M3x"), ("M2", "M0"), ("M1", "M0"), ("M3x", "M1")]
    rng = np.random.default_rng(0)
    boots = {f"{a}-{b}": [] for a, b in pairs}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(n3), len(n3))
        ci = {k: concordance_index_censored(e[i], t[i], r[i])[0] for k, r in risks.items()}
        for a, b in pairs:
            boots[f"{a}-{b}"].append(ci[a] - ci[b])
    delta = {k: [c[k.split("-")[0]] - c[k.split("-")[1]], *map(float, np.percentile(v, [2.5, 97.5]))]
             for k, v in boots.items()}

    # calibration: Breslow baseline from the training data, applied to NHANES III
    calib = {}
    tr_e, tr_t = main_df.dead10.to_numpy().astype(bool), main_df.t10.to_numpy()
    obs = km10(t, e)
    for k in ["M2", "M3x", "M4x"]:
        r_tr = predict({k: models[k]}, main_df)[k]
        br = M.Breslow().fit(r_tr, tr_t, tr_e)
        p = 1 - br.surv(risks[k], [F.HORIZON])[:, 0]
        q = pd.qcut(p, 10, labels=False)
        calib[k] = {"mean_pred": float(p.mean()), "observed": obs,
                    "deciles_pred": [float(p[q == g].mean()) for g in range(10)],
                    "deciles_obs": [km10(t[q == g], e[q == g]) for g in range(10)]}

    # within-era reference: same procedure, 2007-10 held out, mapped markers only
    tr, te = main_df[main_df.era != "2007-10"], main_df[main_df.era == "2007-10"].reset_index(drop=True)
    ref_models = fit_all(tr, {k: fs[k] for k in ["M2", "M3x"]})
    rr = predict(ref_models, te)
    ee, tt = te.dead10.astype(bool).to_numpy(), te.t10.to_numpy()
    ref = {k: float(concordance_index_censored(ee, tt, r)[0]) for k, r in rr.items()}
    ref["M3x-M2"] = ref["M3x"] - ref["M2"]

    shift = {m: {"nhanes3_median": float(n3[m].median()), "train_median": float(main_df[m].median()),
                 "pct_shift": float(100 * (n3[m].median() / main_df[m].median() - 1))}
             for m in fs["M3x"] if m.startswith("lab_")}

    out = {"n": int(len(n3)), "deaths_10y": int(n3.dead10.sum()), "n_train": int(len(main_df)),
           "c": c, "delta": delta, "calibration": calib, "within_era_reference_2007_10": ref,
           "marker_shift": shift, "best_iter": {k: int(b.best_iteration) for k, (b, _) in models.items()},
           "H11_supported": bool(delta["M3x-M2"][1] > 0),
           "H12_supported": bool(delta["M2-M1"][1] > 0)}
    (REP / "nhanes3_external.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["c", "delta", "within_era_reference_2007_10", "H11_supported", "H12_supported"]},
                     indent=1, default=lambda x: round(float(x), 4)))
    print({k: (round(v["mean_pred"], 3), round(v["observed"], 3)) for k, v in calib.items()})
    plot(out)


def plot(out):
    labels = {"M0": "age only", "M1": "PhenoAge*", "M2": "XGB 9 markers", "M3x": "XGB 35 markers",
              "M4x": "XGB 35 + clinical"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ks = list(labels)
    ax.barh(range(len(ks)), [out["c"][k] for k in ks], color=["#999", "#E45756", "#4C78A8", "#4C78A8", "#72B7B2"])
    for i, k in enumerate(ks):
        ax.text(out["c"][k] + 0.001, i, f"{out['c'][k]:.3f}", va="center", fontsize=9)
    ax.set_yticks(range(len(ks)), [labels[k] for k in ks])
    ax.set_xlim(0.80, 0.88)
    ax.set_xlabel("Harrell's C, 10-year mortality, NHANES III")
    ax.set_title(f"Trained on 1999–2018, tested on 1988–94 (n={out['n']:,})\n*PhenoAge was fitted on NHANES III",
                 fontsize=9)
    ax = axes[1]
    for k, col in [("M2", "#4C78A8"), ("M3x", "#F58518"), ("M4x", "#72B7B2")]:
        cl = out["calibration"][k]
        ax.plot(cl["deciles_pred"], cl["deciles_obs"], "o-", color=col, label=f"{labels[k]}")
    ax.plot([0, 0.7], [0, 0.7], "k--", lw=1)
    ax.set_xlabel("predicted 10-year risk")
    ax.set_ylabel("observed 10-year risk (Kaplan–Meier)")
    ax.set_title("Calibration by decile, NHANES III", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nhanes3_external.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
