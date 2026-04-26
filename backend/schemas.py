# D:\www\hate-speech\backend\schemas.py
from pydantic import BaseModel, Field
from typing import List, Literal


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000,
                      examples=["Saya tidak suka dengan orang itu"])
    model_type: Literal['svm', 'mnb', 'cnb', 'indobert'] = Field(
        default="svm",
        description="Pilih model: svm, mnb, cnb, indobert"
    )


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_terms: List[str]
    processed_text: str
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=50)
    model_type: Literal['svm', 'mnb', 'cnb', 'indobert'] = Field(
        default="svm",
        description="Pilih model: svm, mnb, cnb, indobert"
    )


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    n_features: int
    classes: List[str]
    version: str
