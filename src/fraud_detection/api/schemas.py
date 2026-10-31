"""Request / response models for the scoring API."""

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    TransactionID: int
    features: dict[str, float | str | None] = Field(
        description="Raw transaction + identity columns, as in the Kaggle files."
    )


class Prediction(BaseModel):
    TransactionID: int
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_uri: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str
