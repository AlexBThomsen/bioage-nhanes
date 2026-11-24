import numpy as np
import pandas as pd

from bioage.phenoage import mortality_score, phenoage


def person(**kw):
    base = dict(albumin=45, creatinine=80, glucose=5.0, crp=0.1, lymph_pct=30, mcv=90,
                rdw=13, alp=70, wbc=6.0, age=50)
    base.update(kw)
    return pd.DataFrame([base])


def test_healthy_person_close_to_age():
    pa = phenoage(person()).iloc[0]
    assert 35 < pa < 55


def test_age_inverts_mortality_score():
    # PhenoAge maps M back to the age whose Gompertz risk equals M; it should be monotone in age
    ages = phenoage(pd.concat([person(age=a) for a in (30, 50, 70)])).to_numpy()
    assert np.all(np.diff(ages) > 0)


def test_worse_markers_raise_age():
    healthy = phenoage(person()).iloc[0]
    sick = phenoage(person(albumin=35, crp=2.0, rdw=16, glucose=9, wbc=10)).iloc[0]
    assert sick - healthy > 10


def test_mortality_score_is_probability():
    m = mortality_score(person(age=90, rdw=20)).iloc[0]
    assert 0 < m < 1


def test_constants_match_bioage_package():
    # regression: denominator was once mistyped as 0.09165 (should be 0.090165)
    import inspect
    from bioage import phenoage as P
    assert "0.090165" in inspect.getsource(P.phenoage)
    assert abs(P.GAMMA - 0.007692696) < 1e-12

