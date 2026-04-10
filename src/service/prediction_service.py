from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import pickle
import os
import sys
from prometheus_client import Counter, Histogram, Gauge, generate_latest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.sklearn_model import load_model, load_scaler

app = FastAPI(title="Traffic Prediction Service", description="LSTM-based traffic prediction for proactive auto-scaling")

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/lstm_model.pkl')
SCALER_PATH = os.environ.get('SCALER_PATH', 'models/scaler.pkl')

model = None
scaler = None

PREDICTION_COUNTER = Counter('prediction_requests_total', 'Total prediction requests')
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Prediction latency')
PREDICTED_VALUE = Gauge('predicted_traffic_value', 'Latest predicted traffic value')


class PredictionRequest(BaseModel):
    traffic_data: list


class PredictionResponse(BaseModel):
    predictions: list
    confidence: float = None


@app.on_event("startup")
async def load_models():
    global model, scaler
    try:
        model = load_model(MODEL_PATH)
        scaler = load_scaler(SCALER_PATH)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    return {"status": "ok", "message": "Traffic Prediction Service is running"}


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    import time
    start = time.time()
    
    PREDICTION_COUNTER.inc()
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        data = np.array(request.traffic_data).reshape(-1, 1)
        if len(data) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 data points")
        
        scaled_data = scaler.transform(data)
        X = scaled_data[-10:].flatten()
        
        # Use predict_single for single prediction
        if hasattr(model, 'predict_single'):
            scaled_pred = model.predict_single(X)
        else:
            scaled_pred = model.predict(X.reshape(1, -1))[0]
        
        predicted_traffic = scaler.inverse_transform([[scaled_pred]])[0][0]
        
        PREDICTION_LATENCY.observe(time.time() - start)
        PREDICTED_VALUE.set(predicted_traffic)
        
        return PredictionResponse(
            predictions=[float(predicted_traffic)],
            confidence=0.85
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
