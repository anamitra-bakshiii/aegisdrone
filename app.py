import os, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import cloudpickle

app = FastAPI(
    title="AegisDrone v34 Inference Service",
    version="1.0.0",
)
classify_signal_fn = None

class SignalPayload(BaseModel):
    feature_vector: List[float]

class PredictionResponse(BaseModel):
    label: str
    soft_score: float
    threat_level: float
    status: str

@app.on_event("startup")
def load_pipeline():
    global classify_signal_fn
    try:
        import mlflow
        import dagshub
        import os

        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "anamitra1205")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "f33aa2e9d62643fb238ff47abbcdd7ad7e9179ed")

        dagshub.init(
            repo_owner="anamitra1205",
            repo_name="my-first-repo",
            mlflow=True,
        )
        mlflow.set_tracking_uri(
            "https://dagshub.com/anamitra1205/my-first-repo.mlflow"
        )

        print("Downloading model from MLflow registry...")
        model_uri = "models:/AegisDrone_v34_Model/latest"
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)

        def classify_fn(fv):
            out = pyfunc_model.predict(
                np.array(fv, dtype=np.float32).reshape(1, -1)
            )
            return out[0] if isinstance(out, list) else out

        classify_signal_fn = classify_fn
        print("✓ Model loaded from MLflow registry")

    except Exception as e:
        import traceback
        print(f"ERROR loading model: {e}")
        traceback.print_exc()
        # Don't crash — let health endpoint return unhealthy
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