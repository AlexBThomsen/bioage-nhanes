"""Metrics: Harrell/Uno C, time-dependent AUC, integrated Brier, calibration, bootstrap CIs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc, integrated_brier_score)


def point_metrics(y_tr, y_te, risk, S, grid, auc_times):
    """S: survival matrix (n, len(grid)). auc_times must be in grid."""
    out = {"harrell_c": concordance_index_censored(y_te["event"], y_te["time"], risk)[0],
           "uno_c": concordance_index_ipcw(y_tr, y_te, risk, tau=grid[-1])[0]}
    cols = [int(np.argmin(np.abs(grid - t))) for t in auc_times]
    auc, _ = cumulative_dynamic_auc(y_tr, y_te, 1 - S[:, cols], np.asarray(auc_times))
    for t, a in zip(auc_times, auc):
        out[f"auc_{t:g}y"] = a
    out["ibs"] = integrated_brier_score(y_tr, y_te, S, grid)
    return out


def bootstrap(y_tr, y_te, risk, S, grid, auc_times, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y_te)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        try:
            rows.append(point_metrics(y_tr, y_te[i], risk[i], S[i], grid, auc_times))
        except ValueError:      # rare resample where a horizon has no events
            continue
    b = pd.DataFrame(rows)
    return b.quantile(0.025), b.quantile(0.975)


def calibration_table(S_h, time, event, horizon, bins=10):
    """Predicted vs Kaplan-Meier observed risk at `horizon`, by decile of predicted risk."""
    pred = 1 - S_h
    q = pd.qcut(pred, bins, labels=False, duplicates="drop")
    rows = []
    for g in np.unique(q):
        m = q == g
        km = KaplanMeierFitter().fit(time[m], event[m])
        rows.append({"bin": g, "predicted": pred[m].mean(),
                     "observed": 1 - float(km.survival_function_at_times(horizon).iloc[0]), "n": m.sum()})
    return pd.DataFrame(rows)
