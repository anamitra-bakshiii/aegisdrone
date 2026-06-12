import os, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import cloudpickle

app = FastAPI(
    title="AegisDrone v34 Inference Service",
    description="Real-time RF classification for the AegisDrone v34 pipeline.",
    version="1.0.0",
)
classify_signal_fn = None

class SignalPayload(BaseModel):
    feature_vector: List[float] = Field(..., min_items=86, max_items=86)

class PredictionResponse(BaseModel):
    label: str
    soft_score: float
    threat_level: float
    status: str

@app.on_event("startup")
def load_pipeline():
    global classify_signal_fn
    for p in ("classify_signal.pkl", "model_artifacts/classify_signal.pkl"):
        if os.path.exists(p):
            with open(p, "rb") as f:
                classify_signal_fn = cloudpickle.load(f)
            print(f"✓ Pipeline loaded from {p}")
            return
    print("⚠  Pipeline file not found.")

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy" if classify_signal_fn else "unhealthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: SignalPayload):
    if classify_signal_fn is None:
        raise HTTPException(503, "Inference engine not initialised.")
    fv = np.array(payload.feature_vector, dtype=np.float32)
    res = classify_signal_fn(fv)
    return PredictionResponse(
        label=res.get("label", "Unknown"),
        soft_score=float(res.get("soft_score", 0.0)),
        threat_level=float(res.get("threat_score", 0.0)),
        status="processed",
    )
