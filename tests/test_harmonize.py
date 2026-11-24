import numpy as np
import pandas as pd

from bioage.harmonize import map_markers, median_scale, quantile_map


def test_quantile_map_removes_a_scale_shift_and_keeps_order():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.lognormal(3, 0.4, 5000))
    new = pd.Series(rng.lognormal(3, 0.4, 3000) * 0.7)          # lab reads 30% low
    new[5] = np.nan
    out = quantile_map(new, ref)
    assert np.isnan(out[5])
    assert abs(out.median() / ref.median() - 1) < 0.03
    ok = new.notna()
    assert (out[ok].rank() == new[ok].rank()).all()             # ranking is untouched


def test_stratified_mapping_keeps_age_differences():
    rng = np.random.default_rng(1)
    def cohort(n, scale):
        age = rng.uniform(20, 85, n)
        return pd.DataFrame({"age": age, "female": rng.integers(0, 2, n), "x": (10 + 0.2 * age) * scale})
    train, new = cohort(6000, 1.0), cohort(4000, 0.7)
    out = map_markers(new, train, ["x"])
    old, young = out[out.age >= 60].x.median(), out[out.age < 40].x.median()
    t_old, t_young = train[train.age >= 60].x.median(), train[train.age < 40].x.median()
    assert abs(old / t_old - 1) < 0.03 and abs(young / t_young - 1) < 0.03


def test_median_scale():
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
    new = pd.DataFrame({"x": [5.0, 10.0, 15.0]})
    assert median_scale(new, train, ["x"]).x.tolist() == [10.0, 20.0, 30.0]
