# AegisDrone v34 — RF Signal Classification

Live API: https://aegisdrone.onrender.com/docs

## Architecture
- **Model**: LightGBM + CNN + Stacking ensemble
- **API**: FastAPI + Uvicorn
- **Container**: Docker
- **Deployment**: Render
- **CI/CD**: GitHub Actions
- **Experiment Tracking**: MLflow + DagsHub
- **Orchestration**: Kubernetes

## Results
- Detection Recall: 99.9%
- False Alarm Rate: 2.1%
- Latency p95: 65ms
- F1 Score: 0.97

## MLflow Dashboard
https://dagshub.com/anamitra1205/my-first-repo.mlflow

## API Endpoints
- GET /health — service health check
- POST /predict — classify RF signal
