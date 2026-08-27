"""Esquemas Pydantic de entrada/salida del API de inferencia."""

from pydantic import BaseModel, ConfigDict, Field


class SalaryInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(ge=17, le=100, examples=[39])
    workclass: str = Field(examples=["Private"])
    fnlwgt: int = Field(examples=[77516])
    education: str = Field(examples=["Bachelors"])
    education_num: int = Field(alias="education-num", examples=[13])
    marital_status: str = Field(
        alias="marital-status", examples=["Never-married"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(alias="capital-gain", ge=0, examples=[0])
    capital_loss: int = Field(alias="capital-loss", ge=0, examples=[0])
    hours_per_week: int = Field(
        alias="hours-per-week", ge=1, le=99, examples=[40])
    native_country: str = Field(
        alias="native-country", examples=["United-States"])


class PredictionOutput(BaseModel):
    salary_prediction: str
    probability_above_50k: float
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
