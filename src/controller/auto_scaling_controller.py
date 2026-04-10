#!/usr/bin/env python3
import os
import sys
import time
import json
import logging
import subprocess
import argparse
import requests
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decision_engine.decision_engine import DecisionEngine, ThresholdBasedDecisionEngine, NoScalingDecisionEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoScalingController:
    def __init__(self, mode="predictive", 
                 prediction_service_url="http://localhost:8000",
                 check_interval=10,
                 container_capacity=200,
                 min_containers=2,
                 max_containers=10,
                 scale_up_threshold=0.8,
                 scale_down_threshold=0.3,
                 delay_simulation=5,
                 fixed_containers=2):
        self.mode = mode
        self.prediction_service_url = prediction_service_url
        self.check_interval = check_interval
        self.container_capacity = container_capacity
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.delay_simulation = delay_simulation
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.docker_compose_file = "/home/quang/projects/AIOps/docker/docker-compose.yml"
        
        if mode == "predictive":
            self.engine = DecisionEngine(
                prediction_service_url=prediction_service_url,
                max_capacity=max_containers,
                min_capacity=min_containers,
                scale_up_threshold=scale_up_threshold,
                scale_down_threshold=scale_down_threshold,
                container_capacity=container_capacity
            )
        elif mode == "threshold":
            self.engine = ThresholdBasedDecisionEngine(
                scale_up_threshold=scale_up_threshold,
                scale_down_threshold=scale_down_threshold,
                container_capacity=container_capacity,
                min_capacity=min_containers
            )
        elif mode == "no_scaling":
            self.engine = NoScalingDecisionEngine(
                fixed_containers=fixed_containers,
                container_capacity=container_capacity
            )
        
        self.scaling_events = []
        self.metrics = []
    
    def get_current_containers(self):
        try:
            docker_dir = os.path.dirname(self.docker_compose_file)
            result = subprocess.run(
                ["docker", "compose", "-f", self.docker_compose_file, "ps", "--format", "json"],
                cwd=docker_dir, capture_output=True, text=True, timeout=10
            )
            services = [json.loads(line) for line in result.stdout.strip().split('\n') if line]
            webapp_count = sum(1 for s in services if 'webapp' in s.get('Service', ''))
            return max(webapp_count, 1)
        except Exception as e:
            logger.warning(f"Could not get container count: {e}")
            return 2
    
    def get_current_traffic(self):
        """Get current traffic from Prometheus or simulate realistic traffic"""
        import random
        import time as time_module
        
        # Get elapsed time since start
        elapsed = time_module.time() - getattr(self, '_start_time', time_module.time())
        
        phase_traffic = [
            (0, 60, 250),
            (60, 120, 500),
            (120, 180, 800),
            (180, 240, 600),
            (240, 300, 250)
        ]
        
        try:
            response = requests.get(
                "http://localhost:9090/api/v1/query",
                params={"query": "sum(rate(traffic_requests_total[30s]))"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success" and data.get("data", {}).get("result"):
                    traffic = float(data["data"]["result"][0]["value"][1])
                    if traffic > 100:  # Only use real data if substantial
                        return traffic
        except Exception as e:
            pass
        
        # Fallback to simulation phase traffic
        for start, end, base in phase_traffic:
            if start <= elapsed < end:
                return base + random.uniform(-30, 30)
        
        return 100 + random.uniform(-20, 20)
    
    def scale_containers(self, target_count):
        current = self.get_current_containers()
        
        if target_count == current:
            logger.info(f"No scaling needed: current={current}, target={target_count}")
            return False
        
        if self.delay_simulation > 0 and target_count > current:
            logger.info(f"Simulating container startup delay: {self.delay_simulation}s")
            time.sleep(self.delay_simulation)
        
        logger.info(f"Scaling containers: {current} -> {target_count}")
        
        try:
            docker_dir = os.path.dirname(self.docker_compose_file)
            if target_count > current:
                subprocess.run(
                    ["docker", "compose", "-f", self.docker_compose_file, "up", "-d", "--scale", f"webapp={target_count}"],
                    cwd=docker_dir, check=True, timeout=30
                )
            elif target_count < current:
                subprocess.run(
                    ["docker", "compose", "-f", self.docker_compose_file, "up", "-d", "--scale", f"webapp={target_count}"],
                    cwd=docker_dir, check=True, timeout=30
                )
            
            self.scaling_events.append({
                'timestamp': datetime.now().isoformat(),
                'mode': self.mode,
                'current': current,
                'target': target_count,
                'action': 'scale_up' if target_count > current else 'scale_down'
            })
            return True
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return False
    
    def run(self, duration=300):
        logger.info(f"Starting auto scaling controller in {self.mode} mode")
        logger.info(f"Container capacity: {self.container_capacity}")
        logger.info(f"Scale thresholds: up={self.scale_up_threshold}, down={self.scale_down_threshold}")
        
        start_time = time.time()
        self._start_time = start_time
        traffic_buffer = []
        
        while time.time() - start_time < duration:
            current_containers = self.get_current_containers()
            current_traffic = self.get_current_traffic()
            traffic_buffer.append(current_traffic)
            traffic_buffer = traffic_buffer[-10:]  # Need at least 10 for prediction
            
            logger.info(f"Current: traffic={current_traffic:.0f}, containers={current_containers}")
            
            prediction = None
            
            if self.mode == "predictive" and len(traffic_buffer) >= 10:
                prediction = self.engine.get_prediction(traffic_buffer)
                if prediction is not None:
                    required, action, reason = self.engine.make_scaling_decision(
                        prediction, current_containers
                    )
                else:
                    logger.warning("Prediction unavailable, using current traffic")
                    required, action, reason = self.engine.make_scaling_decision(
                        current_traffic, current_containers
                    )
            elif self.mode == "predictive":
                # Not enough data for prediction yet, use current traffic
                required, action, reason = self.engine.make_scaling_decision(
                    current_traffic, current_containers
                )
            elif self.mode == "threshold":
                required, action, reason = self.engine.make_scaling_decision(
                    current_traffic, current_containers
                )
            elif self.mode == "no_scaling":
                # No scaling mode: always use fixed containers
                required, action, reason = self.engine.make_scaling_decision(
                    current_traffic, current_containers
                )
            
            required = max(self.min_containers, min(self.max_containers, required))
            
            pred_value = prediction if (self.mode == "predictive" and prediction is not None) else current_traffic
            
            self.metrics.append({
                'timestamp': time.time() - start_time,
                'mode': self.mode,
                'current_traffic': current_traffic,
                'predicted_traffic': pred_value,
                'current_containers': current_containers,
                'required_containers': required,
                'action': action,
                'reason': reason
            })
            
            if action != "no_change" and self.mode != "no_scaling":
                self.scale_containers(required)
            
            time.sleep(self.check_interval)
        
        self.save_results()
        logger.info("Auto scaling controller finished")
    
    def save_results(self):
        os.makedirs('experiments', exist_ok=True)
        
        with open(f'experiments/{self.mode}_scaling_events.json', 'w') as f:
            json.dump(self.scaling_events, f, indent=2)
        
        with open(f'experiments/{self.mode}_metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Auto Scaling Controller')
    parser.add_argument('--mode', choices=['predictive', 'threshold', 'no_scaling'], default='predictive',
                       help='Scaling mode: predictive (AI-based), threshold (reactive), no_scaling (baseline)')
    parser.add_argument('--fixed-containers', type=int, default=2,
                       help='Fixed number of containers for no_scaling mode')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds')
    parser.add_argument('--interval', type=int, default=10,
                       help='Check interval in seconds')
    parser.add_argument('--min-containers', type=int, default=2)
    parser.add_argument('--max-containers', type=int, default=10)
    parser.add_argument('--capacity', type=int, default=200,
                       help='Container capacity (requests per second)')
    parser.add_argument('--scale-up', type=float, default=0.8,
                       help='Scale up threshold (utilization above this triggers scale up)')
    parser.add_argument('--scale-down', type=float, default=0.3,
                       help='Scale down threshold (utilization below this triggers scale down)')
    parser.add_argument('--delay', type=int, default=5,
                       help='Simulated container startup delay')
    
    args = parser.parse_args()
    
    controller = AutoScalingController(
        mode=args.mode,
        check_interval=args.interval,
        container_capacity=args.capacity,
        min_containers=args.min_containers,
        max_containers=args.max_containers,
        scale_up_threshold=args.scale_up,
        scale_down_threshold=args.scale_down,
        delay_simulation=args.delay,
        fixed_containers=args.fixed_containers
    )
    
    controller.run(duration=args.duration)


if __name__ == "__main__":
    main()
