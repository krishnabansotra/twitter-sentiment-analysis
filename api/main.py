"""FastAPI service.  Run: uvicorn api.main:app --reload"""
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import get_predictor, available_backends

app = FastAPI(title="Twitter/X Sentiment API", version="1.0")


class TweetIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    model: str = "baseline"


class BatchIn(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)
    model: str = "baseline"


def _predictor(name: str):
    if name not in available_backends():
        raise HTTPException(400, f"Model '{name}' not available. Available: {available_backends()}")
    return get_predictor(name)


@app.get("/health")
def health():
    return {"status": "ok", "available_models": available_backends()}


@app.post("/predict")
def predict(t: TweetIn):
    return _predictor(t.model).predict([t.text])[0]


@app.post("/predict_batch")
def predict_batch(b: BatchIn):
    return {"results": _predictor(b.model).predict(b.texts)}
