"""FastAPI router. Mount it in an existing app:

    from bioage.api import router as bioage_router
    app.include_router(bioage_router)

or run standalone for testing:  uvicorn bioage.api:app --reload
"""
from fastapi import APIRouter, FastAPI

from .predict import BioAgeResult, LabReport, load_model, predict

router = APIRouter(prefix="/bioage", tags=["bioage"])


@router.post("/predict", response_model=BioAgeResult)
def predict_route(report: LabReport) -> BioAgeResult:
    return predict(report)


@router.get("/model-info")
def model_info() -> dict:
    b = load_model()
    return {"model": "XGBoost-Cox, feature set C", "features": b["features"],
            "test_c_index": round(b["test_c_index"], 3), "trained_on": b["trained_on"]}


app = FastAPI(title="bioage")
app.include_router(router)
