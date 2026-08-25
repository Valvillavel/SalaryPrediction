"""API REST de inferencia — sirve el modelo Random Forest afinado (`random_forest_tuned.pkl`)."""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src import config
from src.api.schemas import SalaryInput, PredictionOutput, HealthResponse

ml_models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el preprocesador y el modelo una sola vez, al arrancar el servidor."""
    try:
        ml_models["preprocessor"] = joblib.load(
            config.MODELS_DIR / "preprocessor.pkl")
        ml_models["model"] = joblib.load(
            config.MODELS_DIR / "random_forest_tuned.pkl")
    except FileNotFoundError:
        # Permite levantar el servicio igual (útil en tests o CI) aunque falten artefactos;
        # /health y /predict avisarán del problema en vez de tumbar el proceso.
        ml_models.clear()
    yield
    ml_models.clear()


app = FastAPI(
    title="Salary Prediction API",
    description="Predice si el ingreso anual de una persona supera los 50K, según el dataset Adult/Census Income.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_loaded="model" in ml_models)


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: SalaryInput):
    if "model" not in ml_models:
        raise HTTPException(
            status_code=503, detail="Modelo no disponible. Corre `dvc repro` primero.")

    row = pd.DataFrame([payload.model_dump(by_alias=True)])
    X = ml_models["preprocessor"].transform(row)

    proba = float(ml_models["model"].predict_proba(X)[:, 1][0])
    label = config.POSITIVE_LABEL if proba >= 0.5 else " <=50K"

    return PredictionOutput(
        salary_prediction=label.strip(),
        probability_above_50k=round(proba, 4),
        model_used="random_forest_tuned",
    )


@app.post("/predict/batch", response_model=list[PredictionOutput])
def predict_batch(payloads: list[SalaryInput]):
    if "model" not in ml_models:
        raise HTTPException(
            status_code=503, detail="Modelo no disponible. Corre `dvc repro` primero.")

    rows = pd.DataFrame([p.model_dump(by_alias=True) for p in payloads])
    X = ml_models["preprocessor"].transform(rows)
    probas = ml_models["model"].predict_proba(X)[:, 1]

    return [
        PredictionOutput(
            salary_prediction=(config.POSITIVE_LABEL if p >=
                               0.5 else " <=50K").strip(),
            probability_above_50k=round(float(p), 4),
            model_used="random_forest_tuned",
        )
        for p in probas
    ]