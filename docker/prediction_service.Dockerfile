FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

ENV PYTHONPATH=/app
ENV MODEL_PATH=models/lstm_model.pth
ENV SCALER_PATH=models/scaler.pkl

EXPOSE 8000

CMD ["uvicorn", "service.prediction_service:app", "--host", "0.0.0.0", "--port", "8000"]
