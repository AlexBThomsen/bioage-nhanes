import pytest

from bioage.predict import MODEL_PATH

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run scripts/run_models.py first")

HEALTHY = dict(age=45, albumin_g_l=46, creatinine_umol_l=80, glucose_mmol_l=5.0, crp_mg_l=0.5,
               lymphocyte_pct=32, mcv_fl=89, rdw_pct=12.8, alp_u_l=60, wbc_10e9_l=5.5)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from bioage.api import app
    return TestClient(app)


def test_predict_minimal(client):
    r = client.post("/bioage/predict", json=HEALTHY)
    assert r.status_code == 200
    body = r.json()
    assert 25 < body["phenoage"] < 55 and 25 < body["model_bioage"] < 60
    assert "sbp" in body["imputed_fields"]


def test_worse_labs_older(client):
    sick = HEALTHY | dict(albumin_g_l=36, crp_mg_l=20, rdw_pct=16, glucose_mmol_l=9, smoking="current")
    a = client.post("/bioage/predict", json=HEALTHY).json()
    b = client.post("/bioage/predict", json=sick).json()
    assert b["model_bioage"] > a["model_bioage"] + 5
    assert b["phenoage"] > a["phenoage"] + 5


def test_validation_rejects_bad_units(client):
    # albumin in g/dL (4.6) instead of g/L should be rejected, not silently scored
    r = client.post("/bioage/predict", json=HEALTHY | dict(albumin_g_l=4.6))
    assert r.status_code == 422
