"""Esquemas Pydantic de entrada y salida para la API de aml_prediction."""

from typing import List, Optional

from pydantic import BaseModel, Field


class SalaryFeatures(BaseModel):
    """Representa una fila de entrada, con los mismos nombres de columna que salary.csv."""

    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    class Config:
        populate_by_name = True  # Pydantic v2 - acepta tanto alias como nombre real


class PredictionRequest(BaseModel):
    """Payload del endpoint /predict: una lista de personas a evaluar."""
    data: List[SalaryFeatures]


class PredictionResult(BaseModel):
    """Resultado de una predicción individual."""
    index: int
    prediction_code: int
    salary_class: str
    confidence_score: Optional[float] = None
    probabilities_detail: Optional[dict] = None
