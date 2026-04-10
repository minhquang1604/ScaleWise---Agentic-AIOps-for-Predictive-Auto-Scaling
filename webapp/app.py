from flask import Flask, jsonify, request
import time
import random
import logging
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import os

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency', ['method', 'endpoint'])
ACTIVE_REQUESTS = Gauge('active_requests', 'Number of active requests')
CONTAINER_INFO = Gauge('container_info', 'Container information', ['service'])

TRAFFIC_COUNTER = Counter('traffic_requests_total', 'Total requests served')
PROCESSING_TIME = Histogram('request_processing_seconds', 'Request processing time')
CONTAINER_COUNT = Gauge('container_count', 'Number of running containers')

start_time = time.time()


@app.route("/")
def index():
    REQUEST_COUNT.labels(method='GET', endpoint='/', status=200).inc()
    return jsonify({
        "status": "ok",
        "message": "Web service is running",
        "uptime": time.time() - start_time
    })


@app.route("/health")
def health():
    return jsonify({"status": "healthy"})


@app.route("/process", methods=["POST"])
def process():
    with ACTIVE_REQUESTS.track_inprogress():
        start = time.time()
        data = request.get_json() or {}
        delay = data.get("delay", 0)
        
        if delay > 0:
            time.sleep(delay)
        
        # Variable processing time - higher when more concurrent requests
        processing = random.uniform(0.02, 0.15)
        time.sleep(processing)
        
        TRAFFIC_COUNTER.inc()
        PROCESSING_TIME.observe(processing)
        
        duration = time.time() - start
        REQUEST_LATENCY.labels(method='POST', endpoint='/process').observe(duration)
        REQUEST_COUNT.labels(method='POST', endpoint='/process', status=200).inc()
        
        return jsonify({
            "processed": True,
            "processing_time": processing,
            "total_requests": TRAFFIC_COUNTER._value.get()
        })


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {'Content-Type': 'text/plain'}


if __name__ == "__main__":
    CONTAINER_INFO.labels(service='webapp').inc()
    CONTAINER_COUNT.set(1)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
