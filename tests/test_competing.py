import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from bioage.competing import fine_gray_data, weighted_cox


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n)
    t1 = rng.exponential(1 / (0.02 * np.exp(0.5 * x1 + 0.3 * x2)))      # event of interest
    t2 = rng.exponential(1 / (0.05 * np.exp(-0.2 * x1)))                # competing event
    c = rng.uniform(5, 20, n)                                           # censoring
    t = np.minimum.reduce([t1, t2, c])
    cause = np.select([t == t1, t == t2], [1, 2], 0)
    return pd.DataFrame({"x1": x1, "x2": x2.astype(float), "time": t}), cause


def test_matches_lifelines_unweighted():
    d, cause = synthetic()
    d2 = d.assign(start=0.0, stop=d.time, fg_event=(cause == 1).astype(int), w=1.0, sid=np.arange(len(d)))
    own = weighted_cox(d2, ["x1", "x2"])
    lf = CoxPHFitter().fit(d2[["x1", "x2", "stop", "fg_event"]], "stop", "fg_event", robust=True).summary
    assert np.allclose(own["hr"], lf["exp(coef)"], rtol=1e-3)
    assert np.allclose(own["lo_robust"], lf["exp(coef) lower 95%"], rtol=1e-2)


def test_fine_gray_matches_lifelines_weighted_point_estimate():
    d, cause = synthetic()
    fg = fine_gray_data(d, "time", cause)
    own = weighted_cox(fg, ["x1", "x2"])
    lf = CoxPHFitter().fit(fg[["x1", "x2", "start", "stop", "fg_event", "w"]], "stop", "fg_event",
                           weights_col="w", entry_col="start").summary
    assert np.allclose(own["hr"], lf["exp(coef)"], rtol=1e-3)
    # x1 raises the event hazard and lowers the competing one, so its sub-distribution HR must be > 1
    assert own.loc["x1", "hr"] > 1
