import time
import threading
import statistics
import json
import logging
import requests
from prometheus_client import start_http_server, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAFFIC_GAUGE = Gauge('current_traffic', 'Current traffic rate')
PREDICTION_GAUGE = Gauge('predicted_traffic', 'Predicted traffic rate')
CONTAINER_GAUGE = Gauge('container_count', 'Number of containers')
LATENCY_GAUGE = Gauge('request_latency', 'Request latency in seconds')


class MonitoringCollector:
    def __init__(self, webapp_url="http://localhost:5000", prometheus_url="http://localhost:9090"):
        self.webapp_url = webapp_url
        self.prometheus_url = prometheus_url
        self.running = False
        self.metrics_history = []
        self.lock = threading.Lock()
    
    def collect_webapp_metrics(self):
        try:
            response = requests.get(f"{self.webapp_url}/metrics", timeout=5)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            logger.debug(f"Could not collect webapp metrics: {e}")
        return None
    
    def query_prometheus(self, query):
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    return float(data["data"]["result"][0]["value"][1])
        except Exception as e:
            logger.debug(f"Prometheus query failed: {e}")
        return None
    
    def collect(self, duration=60, interval=5):
        self.running = True
        start_time = time.time()
        
        while time.time() - start_time < duration and self.running:
            metric = {
                'timestamp': time.time() - start_time,
                'traffic': self.query_prometheus('rate(traffic_requests_total[1m])') or 0,
                'latency': self.query_prometheus('rate(request_processing_seconds_sum[1m]) / rate(request_processing_seconds_count[1m])') or 0,
                'containers': self.query_prometheus('container_count') or 1
            }
            
            with self.lock:
                self.metrics_history.append(metric)
            
            TRAFFIC_GAUGE.set(metric['traffic'])
            LATENCY_GAUGE.set(metric['latency'])
            CONTAINER_GAUGE.set(metric['containers'])
            
            logger.info(f"Metrics: traffic={metric['traffic']:.2f}, latency={metric['latency']:.4f}, containers={metric['containers']}")
            
            time.sleep(interval)
        
        self.save_metrics()
    
    def save_metrics(self):
        with self.lock:
            with open('experiments/monitoring_metrics.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
    
    def stop(self):
        self.running = False


if __name__ == "__main__":
    collector = MonitoringCollector()
    print("Starting monitoring collector...")
    collector.collect(duration=300)
