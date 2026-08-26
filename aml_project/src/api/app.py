"""API de inferencia para el clasificador de salario (Adult Census Income)."""

import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.model_loader import load_model, load_preprocessor, get_model_metadata, MODEL_NAME
from src.api.schemas import PredictionRequest

# ==========================================
# 1. INICIALIZACIÓN
# ==========================================
app = FastAPI(
    title="API de Predicción de Nivel Salarial",
    description="API MLOps para el clasificador de salary (<=50K / >50K) usando MLflow Model Registry.",
    version="1.0.0",
)

model = None
preprocessor = None
model_version_info = {"version": "Desconocida", "run_id": "Desconocido"}


@app.on_event("startup")
def startup_event():
    """Carga el modelo y el preprocessor de producción al iniciar la API."""
    global model, preprocessor, model_version_info

    model_version_info = get_model_metadata()
    model = load_model()

    if model and model_version_info["run_id"] != "Desconocido":
        preprocessor = load_preprocessor(model_version_info["run_id"])

    if model and preprocessor:
        print(f"✅ [FastAPI] Modelo v{model_version_info['version']} y preprocessor cargados.")
    else:
        print("❌ [FastAPI ERROR] Modelo o preprocessor no disponibles.")


# ==========================================
# 2. ENDPOINTS
# ==========================================
@app.get("/")
def read_root():
    """Chequeo de salud y metadata del modelo en producción."""
    return {
        "status": "Online" if model else "Modelo no disponible",
        "model_name": MODEL_NAME,
        "production_version": model_version_info["version"],
        "run_id": model_version_info["run_id"],
    }


@app.post("/predict")
def predict(payload: PredictionRequest):
    """Predice si el salario es <=50K o >50K para cada registro enviado.
    
    IMPORTANTE: Aplica preprocessor.transform() ANTES de model.predict() porque
    el dataset salary.csv tiene columnas categóricas que deben ser codificadas.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo o el preprocessor no están cargados. Verificá el Model Registry.",
        )

    try:
        # Convertir lista de SalaryFeatures a DataFrame con alias (nombres originales de CSV)
        input_df = pd.DataFrame([item.model_dump(by_alias=True) for item in payload.data])

        # PASO CLAVE: codificar/escalar antes de predecir (categóricas → numéricas)
        input_transformed = preprocessor.transform(input_df)

        # Predicción con el modelo
        predictions = model.predict(input_transformed)
        
        # Probabilidades si el modelo las soporta
        try:
            probabilities = model.predict_proba(input_transformed)
        except AttributeError:
            probabilities = None

        # Formatear resultados
        results = []
        for i, pred in enumerate(predictions):
            label = ">50K" if pred == 1 else "<=50K"
            prob_gt50k = float(probabilities[i][1]) if probabilities is not None else None
            prob_le50k = float(probabilities[i][0]) if probabilities is not None else None
            confidence = float(max(probabilities[i])) if probabilities is not None else None

            results.append({
                "index": i,
                "prediction_code": int(pred),
                "salary_class": label,
                "confidence_score": round(confidence * 100, 2) if confidence else None,
                "probabilities_detail": {
                    "<=50K": round(prob_le50k, 4) if prob_le50k is not None else None,
                    ">50K": round(prob_gt50k, 4) if prob_gt50k is not None else None,
                },
            })

        return {
            "model_metadata": {
                "name": MODEL_NAME,
                "version": model_version_info["version"],
                "run_id": model_version_info["run_id"],
            },
            "total_predictions": len(predictions),
            "results": results,
            "message": "Inferencia completada con éxito.",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la inferencia: {str(e)}")
