"""Request / response models for the scoring API."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_FEATURES = 600  # raw transaction + identity tables have ~430 columns
MAX_BATCH_SIZE = 1000

EXAMPLE_TRANSACTION = {
    "TransactionID": 3663549,
    "features": {
        "TransactionDT": 18403224,
        "TransactionAmt": 31.95,
        "ProductCD": "W",
        "card1": 10409,
        "addr1": 315,
        "P_emaildomain": "gmail.com",
    },
}


class Transaction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_TRANSACTION]})

    TransactionID: int
    features: dict[str, float | int | str | None] = Field(
        max_length=MAX_FEATURES,
        description="Raw transaction + identity columns, as in the Kaggle files.",
    )


TransactionBatch = Annotated[list[Transaction], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class Prediction(BaseModel):
    TransactionID: int
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str
    model_uri: str


class Liveness(BaseModel):
    status: str


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_uri: str  # as configured (MODEL_URI), e.g. an alias
    resolved_model_uri: str | None = None  # the pinned version actually served
    model_version: str | None = None
    model_type: str | None = None
    threshold: float | None = None
    detail: str | None = None  # load error, if any
