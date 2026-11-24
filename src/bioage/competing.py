"""Fine-Gray competing-risks regression via IPCW-weighted Cox (Geskus 2011), with cluster-robust SEs.

lifelines can fit the weighted Cox model but its robust variance is far too slow on the expanded data,
so this module has a small Newton-Raphson fitter that loops over the (few) event times only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def fine_gray_data(d: pd.DataFrame, time: str, cause, grid_step: float = 1.0) -> pd.DataFrame:
    """Expand data so subjects with a competing event (cause==2) stay in the risk set after their death,
    on intervals of `grid_step` years, weighted by G(t)/G(T_i) where G is the censoring survival (KM).
    cause: 0 censored, 1 event of interest, 2 competing event. Adds columns start, stop, fg_event, w, sid."""
    cause = np.asarray(cause)
    d = d.reset_index(drop=True).assign(sid=np.arange(len(d)))
    km = KaplanMeierFitter().fit(d[time], (cause == 0).astype(int))
    G = lambda t: np.clip(km.survival_function_at_times(t).to_numpy(), 1e-3, 1)  # noqa: E731
    tmax = float(d[time].max())
    base = d.copy()
    base["start"], base["stop"] = 0.0, base[time].astype(float)
    base["fg_event"], base["w"] = (cause == 1).astype(int), 1.0

    comp = d[cause == 2].reset_index(drop=True)
    Ti = comp[time].to_numpy(float)
    n_int = np.ceil((tmax - Ti) / grid_step).astype(int)
    keep = n_int > 0
    comp, Ti, n_int = comp[keep].reset_index(drop=True), Ti[keep], n_int[keep]
    rep = np.repeat(np.arange(len(comp)), n_int)
    k = np.concatenate([np.arange(n) for n in n_int]) if len(n_int) else np.array([], int)
    starts = Ti[rep] + k * grid_step
    ext = comp.iloc[rep].reset_index(drop=True)
    ext["start"], ext["stop"], ext["fg_event"] = starts, np.minimum(starts + grid_step, tmax), 0
    ext["w"] = G(starts) / G(Ti)[rep]
    out = pd.concat([base, ext], ignore_index=True)
    return out[out["stop"] > out["start"]].reset_index(drop=True)


def weighted_cox(df: pd.DataFrame, covs: list[str], start="start", stop="stop", event="fg_event",
                 weight="w", cluster="sid", max_iter=30, tol=1e-9) -> pd.DataFrame:
    """Weighted Cox PH with left truncation (counting process), Breslow ties.
    Returns HR with model-based and cluster-robust (sandwich) 95% CIs."""
    X = df[covs].to_numpy(float)
    mu = X.mean(0)
    X = X - mu                                     # centering for numerical stability
    w = df[weight].to_numpy(float)
    s0, s1 = df[start].to_numpy(float), df[stop].to_numpy(float)
    ev = df[event].to_numpy(int) == 1
    times = np.unique(s1[ev])
    risk_sets = [np.flatnonzero((s0 < t) & (s1 >= t)) for t in times]
    ev_sets = [np.flatnonzero(ev & (s1 == t)) for t in times]

    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(max_iter):
        r = w * np.exp(X @ beta)
        grad, H = np.zeros(p), np.zeros((p, p))
        for R, D in zip(risk_sets, ev_sets):
            rR, XR = r[R], X[R]
            S0 = rR.sum()
            xbar = rR @ XR / S0
            S2 = (XR * rR[:, None]).T @ XR / S0
            dw = w[D].sum()
            grad += w[D] @ X[D] - dw * xbar
            H += dw * (S2 - np.outer(xbar, xbar))
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < tol:
            break

    # score residuals (Lin & Wei), summed within cluster
    r = w * np.exp(X @ beta)
    U = np.zeros_like(X)
    for R, D in zip(risk_sets, ev_sets):
        rR, XR = r[R], X[R]
        S0 = rR.sum()
        xbar = rR @ XR / S0
        dw = w[D].sum()
        U[D] += w[D][:, None] * (X[D] - xbar)
        U[R] -= (rR * dw / S0)[:, None] * (XR - xbar)
    Uc = pd.DataFrame(U).groupby(df[cluster].to_numpy()).sum().to_numpy()
    Hinv = np.linalg.inv(H)
    V_rob = Hinv @ (Uc.T @ Uc) @ Hinv
    se_m, se_r = np.sqrt(np.diag(Hinv)), np.sqrt(np.diag(V_rob))
    return pd.DataFrame({
        "coef": beta, "hr": np.exp(beta),
        "lo_robust": np.exp(beta - 1.96 * se_r), "hi_robust": np.exp(beta + 1.96 * se_r),
        "lo_model": np.exp(beta - 1.96 * se_m), "hi_model": np.exp(beta + 1.96 * se_m),
    }, index=covs)
