"""FastAPI app that serves the trained model."""
import os
from contextlib import asynccontextmanager
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and threshold once, when the server starts."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"{MODEL_PATH} not found. Run `python train.py` first.")
    artifact = joblib.load(MODEL_PATH)
    app.state.model = artifact["model"]
    app.state.threshold = artifact["threshold"]
    yield


app = FastAPI(title="Model Serving API", version="0.1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Input must contain exactly 10 numeric features."""
    features: List[float] = Field(..., min_length=10, max_length=10)


class PredictResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)


@app.get("/health")
def health() -> dict:
    """Simple check that the server is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    """Return the class-1 probability and the final 0/1 prediction."""
    state = request.app.state
    try:
        prob = float(state.model.predict_proba([req.features])[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
    return PredictResponse(probability=prob, prediction=int(prob >= state.threshold))