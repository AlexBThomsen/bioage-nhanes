"""Pre-registered (PREREGISTRATION.md, H13-H14): fix NHANES III calibration by mapping markers onto the training
scale, without using NHANES III outcomes.

Usage:  PYTHONPATH=src python scripts/recalibration.py
Writes: reports/recalibration.json, reports/figures/recalibration.png
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
from sksurv.metrics import concordance_index_censored as cindex  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import full_panel as F  # noqa: E402
import nhanes3_external as X  # noqa: E402
from bioage import models as M  # noqa: E402
from bioage.harmonize import map_markers, median_scale  # noqa: E402
from bioage.phenoage import MARKERS  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"


def main():
    train, n3 = F.load(), X.load_n3()
    fs = X.feature_sets(train)
    extras = [m for m in fs["M3x"] if m.startswith("lab_")]
    models = X.fit_all(train, {k: fs[k] for k in ["M2", "M3x"]})
    breslow = {}
    for k, (b, dm) in models.items():
        r = b.predict(dm(train), output_margin=True, iteration_range=(0, b.best_iteration + 1))
        breslow[k] = M.Breslow().fit(r, train.t10.to_numpy(), train.dead10.to_numpy().astype(bool))

    variants = {
        "unmapped": n3,
        "stratified quantile map (primary)": map_markers(n3, train, extras),
        "unstratified quantile map": map_markers(n3, train, extras, stratified=False),
        "median-ratio scaling": median_scale(n3, train, extras),
        "stratified map, all 35 markers": map_markers(n3, train, extras + MARKERS),
    }
    e, t = n3.dead10.astype(bool).to_numpy(), n3.t10.to_numpy()
    obs = X.km10(t, e)

    def score(k, d):
        b, dm = models[k]
        r = b.predict(dm(d), output_margin=True, iteration_range=(0, b.best_iteration + 1))
        p = 1 - breslow[k].surv(r, [F.HORIZON])[:, 0]
        q = pd.qcut(p, 10, labels=False)
        dp = np.array([p[q == g].mean() for g in range(10)])
        do = np.array([X.km10(t[q == g], e[q == g]) for g in range(10)])
        return r, {"c": float(cindex(e, t, r)[0]), "mean_pred": float(p.mean()), "pred_over_obs": float(p.mean() / obs),
                   "ici_deciles": float(np.mean(np.abs(dp - do))), "deciles_pred": dp.tolist(), "deciles_obs": do.tolist()}

    r_m2, m2 = score("M2", n3)
    res, risks = {}, {}
    for name, d in variants.items():
        risks[name], res[name] = score("M3x", d)
        print(f"{name:36s} C={res[name]['c']:.4f} pred={res[name]['mean_pred']:.3f} obs={obs:.3f} "
              f"ICI={res[name]['ici_deciles']:.4f}", flush=True)

    prim = "stratified quantile map (primary)"
    rng = np.random.default_rng(0)
    d_m2, d_unm = [], []
    for _ in range(1000):
        i = rng.integers(0, len(n3), len(n3))
        cp = cindex(e[i], t[i], risks[prim][i])[0]
        d_m2.append(cp - cindex(e[i], t[i], r_m2[i])[0])
        d_unm.append(cp - cindex(e[i], t[i], risks["unmapped"][i])[0])
    ci = lambda pt, b: [float(pt), *map(float, np.percentile(b, [2.5, 97.5]))]  # noqa: E731
    out = {"observed_10y": obs, "M2_unmapped": m2, "M3x": res,
           "primary_minus_M2": ci(res[prim]["c"] - m2["c"], d_m2),
           "primary_minus_unmapped": ci(res[prim]["c"] - res["unmapped"]["c"], d_unm)}
    out["H13_supported"] = bool(abs(res[prim]["mean_pred"] / obs - 1) <= 0.10
                                and res[prim]["c"] >= res["unmapped"]["c"] - 0.002)
    out["H14_supported"] = bool(out["primary_minus_M2"][1] > 0)
    (REP / "recalibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ["primary_minus_M2", "primary_minus_unmapped", "H13_supported", "H14_supported"]},
                     indent=1))
    print(f"M2: C={m2['c']:.4f} pred={m2['mean_pred']:.3f} ICI={m2['ici_deciles']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for name, col in [("unmapped", "#E45756"), (prim, "#4C78A8"), ("median-ratio scaling", "#72B7B2")]:
        ax.plot(res[name]["deciles_pred"], res[name]["deciles_obs"], "o-", color=col, label=name)
    ax.plot([0, 0.85], [0, 0.85], "k--", lw=1)
    ax.set_xlabel("predicted 10-year risk (35-marker model)")
    ax.set_ylabel("observed 10-year risk (Kaplan–Meier)")
    ax.set_title("Calibration in NHANES III before and after mapping", fontsize=9)
    ax.legend(fontsize=8)
    ax = axes[1]
    names = list(variants)
    vals = [res[n]["mean_pred"] for n in names] + [m2["mean_pred"]]
    labels = names + ["9-marker model (reference)"]
    ax.barh(range(len(vals)), vals, color=["#E45756", "#4C78A8", "#9ecae9", "#72B7B2", "#bab0ac", "#999"])
    ax.axvline(obs, color="k", lw=1.2)
    ax.axvspan(obs * 0.9, obs * 1.1, color="k", alpha=0.08)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.1%}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals)), labels, fontsize=8)
    ax.set_xlabel("mean predicted 10-year mortality")
    ax.set_title(f"Line: observed {obs:.1%}; shaded band: pre-registered ±10% target", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "recalibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
