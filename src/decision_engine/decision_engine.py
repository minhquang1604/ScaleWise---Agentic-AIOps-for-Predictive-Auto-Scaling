import os
import sys
import time
import logging
import requests
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.data_preprocessing import WINDOW_SIZE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DecisionEngine:
    def __init__(self, prediction_service_url="http://localhost:8000", 
                 max_capacity=1000, min_capacity=2, 
                 scale_up_threshold=0.8, scale_down_threshold=0.5,
                 container_capacity=200):
        self.prediction_service_url = prediction_service_url
        self.max_capacity = max_capacity
        self.min_capacity = min_capacity
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.container_capacity = container_capacity
        self.traffic_history = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        self.scaling_history = []
    
    def get_prediction(self, traffic_data):
        try:
            response = requests.post(
                f"{self.prediction_service_url}/predict",
                json={"traffic_data": traffic_data},
                timeout=5
            )
            if response.status_code == 200:
                return response.json()["predictions"][0]
        except Exception as e:
            logger.error(f"Prediction service error: {e}")
        return None
    
    def calculate_required_containers(self, predicted_traffic):
        return max(self.min_capacity, min(
            self.max_capacity,
            int(np.ceil(predicted_traffic / self.container_capacity))
        ))
    
    def make_scaling_decision(self, predicted_traffic, current_containers):
        self.prediction_history.append({
            'timestamp': time.time(),
            'predicted': predicted_traffic
        })
        
        if predicted_traffic is None:
            return current_containers, "no_change", "prediction_unavailable"
        
        required_containers = self.calculate_required_containers(predicted_traffic)
        utilization = predicted_traffic / (current_containers * self.container_capacity)
        
        # PROACTIVE: Scale based on PREDICTED traffic - if predicted needs more containers, scale NOW
        if required_containers > current_containers:
            action = "scale_up"
            reason = f"PROACTIVE: pred_traffic={predicted_traffic:.0f} needs {required_containers} containers, have {current_containers} -> SCALE BEFORE OVERLOAD"
        elif required_containers < current_containers and utilization < self.scale_down_threshold and current_containers > self.min_capacity:
            action = "scale_down"
            reason = f"PROACTIVE: pred_traffic={predicted_traffic:.0f} needs only {required_containers}, utilization={utilization:.2f} < {self.scale_down_threshold}"
        else:
            action = "no_change"
            reason = f"PROACTIVE: pred_traffic={predicted_traffic:.0f} needs {required_containers} containers, current={current_containers} OK"
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'predicted_traffic': predicted_traffic,
            'current_containers': current_containers,
            'required_containers': required_containers,
            'action': action,
            'reason': reason
        })
        
        logger.info(f"Predictive Decision: pred_traffic={predicted_traffic:.0f}, containers={current_containers}, "
                   f"pred_util={utilization:.2f}, action={action}")

        return required_containers, action, reason
    
    def get_current_containers(self):
        try:
            result = os.popen("docker compose ps --format json 2>/dev/null | python3 -c \"import sys,json; print(len([l for l in map(json.loads, sys.stdin) if 'webapp' in l.get('Service','')]))\" 2>/dev/null").read().strip()
            if result.isdigit():
                return int(result)
        except:
            pass
        return 1
    
    def get_metrics(self):
        return {
            'traffic_history': list(self.traffic_history),
            'prediction_history': list(self.prediction_history),
            'scaling_history': list(self.scaling_history[-20:])
        }


class ThresholdBasedDecisionEngine:
    def __init__(self, scale_up_threshold=0.8, scale_down_threshold=0.3, container_capacity=200, min_capacity=2):
        """Threshold-based: ONLY scale after overload is detected (reactive)"""
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.container_capacity = container_capacity
        self.min_capacity = min_capacity
        self.scaling_history = []
    
    def make_scaling_decision(self, current_traffic, current_containers):
        utilization = current_traffic / (current_containers * self.container_capacity)
        
        required = max(self.min_capacity, int(np.ceil(current_traffic / self.container_capacity)))
        
        # REACTIVE: Only scale AFTER overload is detected (util > threshold)
        if utilization > self.scale_up_threshold and required > current_containers:
            action = "scale_up"
            reason = f"REACTIVE: util={utilization:.2f} > {self.scale_up_threshold} -> ALREADY OVERLOADED, scaling NOW"
        elif utilization < self.scale_down_threshold and required < current_containers and current_containers > self.min_capacity:
            action = "scale_down"
            reason = f"REACTIVE: util={utilization:.2f} < {self.scale_down_threshold} -> scaling down"
        else:
            action = "no_change"
            reason = f"REACTIVE: util={utilization:.2f} in range [{self.scale_down_threshold}, {self.scale_up_threshold}], waiting for overload..."
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'current_traffic': current_traffic,
            'current_containers': current_containers,
            'required_containers': required,
            'utilization': utilization,
            'action': action,
            'reason': reason
        })
        
        logger.info(f"Threshold Decision: traffic={current_traffic:.0f}, containers={current_containers}, "
                   f"util={utilization:.2f}, action={action}")

        return required, action, reason


class NoScalingDecisionEngine:
    """No scaling mode - baseline with fixed containers"""
    def __init__(self, fixed_containers=2, container_capacity=200):
        self.fixed_containers = fixed_containers
        self.container_capacity = container_capacity
        self.scaling_history = []
    
    def make_scaling_decision(self, current_traffic, current_containers):
        """Always return fixed containers - no scaling"""
        utilization = current_traffic / (self.fixed_containers * self.container_capacity)
        
        required = self.fixed_containers
        action = "no_change"
        reason = f"no_scaling_mode_fixed_{self.fixed_containers}_containers"
        
        self.scaling_history.append({
            'timestamp': time.time(),
            'current_traffic': current_traffic,
            'current_containers': current_containers,
            'required_containers': required,
            'action': action,
            'reason': reason
        })
        
        logger.info(f"No Scaling Decision: current_traffic={current_traffic:.0f}, "
                   f"fixed={self.fixed_containers}, utilization={utilization:.2f}")

        return required, action, reason


if __name__ == "__main__":
    engine = DecisionEngine()
    print("Decision Engine initialized")
    print(f"Scale up threshold: {engine.scale_up_threshold}")
    print(f"Scale down threshold: {engine.scale_down_threshold}")
    print(f"Container capacity: {engine.container_capacity}")
