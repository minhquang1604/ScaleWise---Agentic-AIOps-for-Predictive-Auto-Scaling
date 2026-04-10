#!/usr/bin/env python3
import time
import random
import argparse
import threading
import statistics
from datetime import datetime
import concurrent.futures

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TrafficGenerator:
    def __init__(self, url="http://localhost:5000", 
                 base_rps=10,
                 spike_rps=50,
                 spike_probability=0.1,
                 spike_duration=10,
                 workers=5,
                 delay_range=(0.05, 0.2),
                 multiplier=1):
        self.url = url
        self.base_rps = base_rps
        self.spike_rps = spike_rps
        self.spike_probability = spike_probability
        self.spike_duration = spike_duration
        self.workers = workers
        self.delay_range = delay_range
        self.multiplier = multiplier
        self.running = False
        self.request_count = 0
        self.response_times = []
        self.errors = 0
        self.lock = threading.Lock()
    
    def send_request(self):
        try:
            start = time.time()
            response = requests.post(f"{self.url}/process", 
                                   json={"delay": random.uniform(self.delay_range[0], self.delay_range[1])},
                                   timeout=10)
            duration = time.time() - start
            
            with self.lock:
                self.request_count += 1
                self.response_times.append(duration)
                if response.status_code != 200:
                    self.errors += 1
            
            return response.status_code == 200
        except Exception as e:
            with self.lock:
                self.errors += 1
            return False
    
    def generate_traffic(self, duration, rps):
        interval = 1.0 / rps if rps > 0 else 1.0
        end_time = time.time() + duration
        
        while time.time() < end_time and self.running:
            self.send_request()
            time.sleep(interval)
    
    def generate_spike_traffic(self, duration, rps, workers):
        """Generate concurrent traffic for spike - ENHANCED for clearer results"""
        end_time = time.time() + duration
        
        while time.time() < end_time and self.running:
            # Burst of concurrent requests to trigger overload
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.send_request) for _ in range(workers)]
                concurrent.futures.wait(futures)
            # Short pause between bursts
            time.sleep(max(0.1, 1.0 / rps))
    
    def run(self, duration=60, stress_test=False):
        if not REQUESTS_AVAILABLE:
            print("requests library not available")
            return
        
        self.running = True
        print(f"Starting traffic generation for {duration}s")
        print(f"Base RPS: {self.base_rps}, Spike RPS: {self.spike_rps}")
        if stress_test:
            print(f"Stress Test Mode: {self.workers} concurrent workers")
        
        start_time = time.time()
        
        if stress_test:
            # ENHANCED stress test: Create clear overload scenarios
            # Phase 1 (0-60s): Normal traffic - should be handled easily
            # Phase 2 (60-120s): Spike - triggers overload in no_scaling
            # Phase 3 (120-180s): Extreme spike - shows difference between threshold vs predictive
            # Phase 4 (180-240s): Another spike
            # Phase 5 (240-300s): Recovery
            
            phases = [
                (0, 60, "NORMAL", 200, 20),
                (60, 120, "SPIKE-1", 500, 40),
                (120, 180, "EXTREME", 800, 60),
                (180, 240, "SPIKE-2", 600, 50),
                (240, 300, "COOL DOWN", 200, 20)
            ]
            
            # Apply multiplier for 10x traffic
            multiplier = getattr(self, 'multiplier', 1)
            if multiplier > 1:
                phases = [(start, end, name, rps * multiplier, workers * multiplier) 
                         for start, end, name, rps, workers in phases]
            
            for phase_start, phase_end, phase_name, phase_rps, phase_workers in phases:
                elapsed = time.time() - start_time
                if elapsed < phase_start:
                    time.sleep(phase_start - elapsed)
                
                remaining = min(phase_end - int(time.time() - start_time), duration)
                if remaining <= 0:
                    break
                    
                print(f"[{int(time.time()-start_time)}s] {phase_name}: RPS={phase_rps}, workers={phase_workers}")
                self.generate_spike_traffic(min(remaining, phase_end - phase_start), phase_rps, phase_workers)
        else:
            # Normal mode with spikes
            while time.time() - start_time < duration and self.running:
                is_spike = random.random() < self.spike_probability
                rps = self.spike_rps if is_spike else self.base_rps
                
                if is_spike:
                    print(f"Spike: RPS={rps}")
                
                self.generate_traffic(min(10, duration - (time.time() - start_time)), rps)
        
        self.running = False
        self.print_stats()
    
    def print_stats(self):
        with self.lock:
            print("\n=== Traffic Generation Statistics ===")
            print(f"Total requests: {self.request_count}")
            print(f"Errors: {self.errors}")
            if self.response_times:
                print(f"Avg response time: {statistics.mean(self.response_times):.4f}s")
                print(f"Min response time: {min(self.response_times):.4f}s")
                print(f"Max response time: {max(self.response_times):.4f}s")
    
    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description='Traffic Generator')
    parser.add_argument('--url', default='http://localhost:5000',
                       help='Target URL')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds')
    parser.add_argument('--base-rps', type=int, default=10,
                       help='Base requests per second')
    parser.add_argument('--spike-rps', type=int, default=50,
                       help='Spike requests per second')
    parser.add_argument('--spike-prob', type=float, default=0.1,
                       help='Spike probability')
    parser.add_argument('--stress', action='store_true',
                       help='Enable stress test mode with extreme spikes')
    parser.add_argument('--workers', type=int, default=20,
                       help='Number of concurrent workers for stress test')
    parser.add_argument('--multiplier', type=int, default=1,
                       help='Traffic multiplier (e.g., 10 for 10x traffic)')
    
    args = parser.parse_args()
    
    generator = TrafficGenerator(
        url=args.url,
        base_rps=args.base_rps,
        spike_rps=args.spike_rps,
        spike_probability=args.spike_prob,
        workers=args.workers,
        multiplier=args.multiplier
    )
    
    try:
        generator.run(args.duration, stress_test=args.stress)
    except KeyboardInterrupt:
        generator.stop()


if __name__ == "__main__":
    main()
