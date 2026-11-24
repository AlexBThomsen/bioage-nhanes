"""Survival models behind one interface.

Every model: fit(train_df, val_df) ; risk(df) -> higher = worse ; surv(df, times_years) -> (n, len(times)).
Models with a log-hazard output use a Breslow baseline estimated on train to get survival curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb  # must be imported before torch: two OpenMP runtimes segfault on macOS
import torch  # noqa: E402
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from . import phenoage as pa

PHENO = pa.MARKERS
FEATURE_SETS = {
    "A": PHENO + ["age"],
    "B": PHENO + ["age", "sbp", "pulse_pressure", "on_bp_meds"],   # DBP = SBP - PP, so it's implied
    "C": PHENO + ["age", "sbp", "pulse_pressure", "on_bp_meds",
                  "total_chol", "hdl", "hba1c", "bmi", "waist", "current_smoker", "ever_smoker", "female"],
}
LOG_FEATURES = {"crp"}


def y_of(df: pd.DataFrame):
    return Surv.from_arrays(df["dead"].astype(bool).to_numpy(), df["time_years"].to_numpy())


class Prep:
    """log-transform skewed markers, median-impute, standardize. Fitted on train only."""

    def __init__(self, features):
        self.features = list(features)

    def _raw(self, df):
        X = df[self.features].astype(float).copy()
        for c in LOG_FEATURES & set(X):
            X[c] = np.log(X[c].clip(lower=0.01))
        return X

    def fit(self, df):
        X = self._raw(df)
        self.med = X.median()
        X = X.fillna(self.med)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1)
        return self

    def transform(self, df) -> np.ndarray:
        X = self._raw(df).fillna(self.med)
        return ((X - self.mu) / self.sd).to_numpy(np.float32)


class Breslow:
    def fit(self, risk, time, event):
        order = np.argsort(time)
        t, e, r = time[order], event[order], np.exp(risk[order] - risk.max())
        self.shift = risk.max()
        at_risk = np.cumsum(r[::-1])[::-1]                 # sum exp(r) over people with T >= t_i
        uniq = np.unique(t[e])
        first = np.searchsorted(t, uniq, side="left")
        d = np.array([e[(t == u)].sum() for u in uniq])
        self.times = uniq
        self.H0 = np.cumsum(d / at_risk[first])
        return self

    def cumhaz0(self, times):
        idx = np.searchsorted(self.times, times, side="right") - 1
        return np.where(idx >= 0, self.H0[np.clip(idx, 0, None)], 0.0)

    def surv(self, risk, times):
        return np.exp(-np.outer(np.exp(risk - self.shift), self.cumhaz0(np.asarray(times))))


class LogHazardModel:
    name = "base"

    def __init__(self, features):
        self.prep = Prep(features)

    def fit(self, tr, va):
        self.prep.fit(tr)
        self._fit(self.prep.transform(tr), y_of(tr), self.prep.transform(va), y_of(va))
        r = self.risk(tr)
        self.base = Breslow().fit(r, tr["time_years"].to_numpy(), tr["dead"].astype(bool).to_numpy())
        return self

    def risk(self, df):
        return self._risk(self.prep.transform(df))

    def surv(self, df, times):
        return self.base.surv(self.risk(df), times)


class PhenoAgeModel:
    """Published formula, nothing is fitted."""
    name = "PhenoAge"

    def __init__(self, features=None):
        pass

    def fit(self, tr, va):
        return self

    def risk(self, df):
        return pa.linear_predictor(df).to_numpy()

    def surv(self, df, times):
        xb = self.risk(df)[:, None]
        m = np.asarray(times)[None, :] * 12
        return np.exp(-np.exp(xb) * (np.exp(pa.GAMMA * m) - 1) / pa.GAMMA)


class Cox(LogHazardModel):
    name = "Cox"

    def _fit(self, X, y, Xv, yv):
        self.m = CoxPHSurvivalAnalysis(alpha=1e-4, ties="efron").fit(X, y)

    def _risk(self, X):
        return self.m.predict(X)


class Coxnet(LogHazardModel):
    name = "Coxnet"

    def _fit(self, X, y, Xv, yv):
        self.m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.001, n_alphas=40).fit(X, y)
        scores = [concordance_index_censored(yv["event"], yv["time"], self.m.predict(Xv, alpha=a))[0]
                  for a in self.m.alphas_]
        self.alpha = self.m.alphas_[int(np.argmax(scores))]
        self.coef = pd.Series(self.m.coef_[:, int(np.argmax(scores))], index=self.prep.features)

    def _risk(self, X):
        return self.m.predict(X, alpha=self.alpha)


class RSF:
    name = "RSF"

    def __init__(self, features):
        self.prep = Prep(features)

    def fit(self, tr, va):
        self.prep.fit(tr)
        X, y, Xv, yv = self.prep.transform(tr), y_of(tr), self.prep.transform(va), y_of(va)
        best = None
        for leaf in (30, 80):
            m = RandomSurvivalForest(n_estimators=200, min_samples_leaf=leaf, max_features="sqrt",
                                     max_samples=0.5, n_jobs=-1, random_state=0).fit(X, y)
            c = concordance_index_censored(yv["event"], yv["time"], m.predict(Xv))[0]
            if best is None or c > best[0]:
                best = (c, leaf, m)
        _, self.leaf, self.m = best
        return self

    def risk(self, df):
        return self.m.predict(self.prep.transform(df))

    def surv(self, df, times):
        S = self.m.predict_survival_function(self.prep.transform(df), return_array=True)
        idx = np.searchsorted(self.m.unique_times_, times, side="right") - 1
        return np.where(idx >= 0, S[:, np.clip(idx, 0, None)], 1.0)


class XGBCox(LogHazardModel):
    name = "XGBoost"

    @staticmethod
    def _dm(X, y):
        label = np.where(y["event"], y["time"], -y["time"])
        return xgb.DMatrix(X, label=label)

    def _fit(self, X, y, Xv, yv):
        params = dict(objective="survival:cox", eval_metric="cox-nloglik", eta=0.03, max_depth=3,
                      subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, seed=0)
        dv = self._dm(Xv, yv)
        self.m = xgb.train(params, self._dm(X, y), num_boost_round=3000, evals=[(dv, "val")],
                           early_stopping_rounds=100, verbose_eval=False)

    def _risk(self, X):
        return self.m.predict(xgb.DMatrix(X), output_margin=True,
                              iteration_range=(0, self.m.best_iteration + 1))


def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    r, e = risk[order], event[order]
    log_cum = torch.logcumsumexp(r, dim=0)
    return -((r - log_cum) * e).sum() / e.sum().clamp(min=1)


class DeepSurv(LogHazardModel):
    name = "DeepSurv"

    def _fit(self, X, y, Xv, yv):
        torch.manual_seed(0)
        d = X.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 32), torch.nn.SiLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 32), torch.nn.SiLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1))
        opt = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-3)
        T = lambda a: torch.tensor(np.array(a, dtype=np.float32))  # noqa: E731
        Xt, tt, et = T(X), T(y["time"]), T(y["event"])
        Xvt, tv, ev = T(Xv), T(yv["time"]), T(yv["event"])
        best, best_state, patience = np.inf, None, 0
        n = len(Xt)
        for epoch in range(300):
            self.net.train()
            perm = torch.randperm(n)
            for b in range(0, n, 2048):   # mini-batches of 2048: risk sets are approximated within batch
                idx = perm[b:b + 2048]
                opt.zero_grad()
                loss = cox_loss(self.net(Xt[idx]).squeeze(-1), tt[idx], et[idx])
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = cox_loss(self.net(Xvt).squeeze(-1), tv, ev).item()
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= 20:
                    break
        self.net.load_state_dict(best_state)
        self.epochs = epoch + 1

    def _risk(self, X):
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X)).squeeze(-1).numpy().astype(float)


MODELS = [PhenoAgeModel, Cox, Coxnet, RSF, XGBCox, DeepSurv]


class AgeReference:
    """Age-only Cox on train. Maps a 10-year death risk back to 'the age with that same risk'."""

    def fit(self, tr, horizon=10.0):
        X = tr[["age"]].to_numpy(float)
        m = CoxPHSurvivalAnalysis().fit(X, y_of(tr))
        self.b = float(m.coef_[0])
        base = Breslow().fit(X[:, 0] * self.b, tr["time_years"].to_numpy(), tr["dead"].astype(bool).to_numpy())
        self.logH = np.log(base.cumhaz0(np.array([horizon]))[0]) - base.shift
        self.horizon = horizon
        return self

    def bio_age(self, risk10):
        r = np.clip(risk10, 1e-6, 1 - 1e-9)
        return (np.log(-np.log(1 - r)) - self.logH) / self.b
