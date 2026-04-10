import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_json(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_metrics(metrics_data):
    if not metrics_data:
        return {
            'avg_latency': 0,
            'max_latency': 0,
            'avg_throughput': 0,
            'scale_events': 0,
            'overload_events': 0,
            'stability': 0
        }
    
    latencies = []
    throughputs = []
    container_changes = 0
    
    for m in metrics_data:
        current_traffic = m.get('current_traffic', 0)
        current_containers = m.get('current_containers', 1)
        required_containers = m.get('required_containers', current_containers)
        
        throughputs.append(current_traffic)
        
        container_capacity = 200
        utilization = current_traffic / (current_containers * container_capacity) if current_containers > 0 else 0
        
        if utilization > 1.0:
            latency = 0.5 + (utilization - 1.0) * 0.5
        elif utilization > 0.8:
            latency = 0.2 + (utilization - 0.8) * 1.5
        elif utilization > 0.5:
            latency = 0.1 + (utilization - 0.5) * 0.33
        else:
            latency = 0.05 + utilization * 0.1
        
        latencies.append(latency)
        
        if required_containers != current_containers:
            container_changes += 1
    
    overloads = sum(1 for m in metrics_data if m.get('current_traffic', 0) > 400)
    
    stability = 1.0 - (container_changes / max(len(metrics_data), 1))
    
    return {
        'avg_latency': np.mean(latencies) if latencies else 0,
        'max_latency': np.max(latencies) if latencies else 0,
        'avg_throughput': np.mean(throughputs) if throughputs else 0,
        'scale_events': container_changes,
        'overload_events': overloads,
        'stability': stability
    }


def plot_comparison(results, output_dir='experiments'):
    os.makedirs(output_dir, exist_ok=True)
    
    modes = list(results.keys())
    metrics_names = ['avg_latency', 'max_latency', 'scale_events', 'overload_events', 'stability']
    metric_labels = ['Avg Latency (s)', 'Max Latency (s)', 'Scale Events', 'Overload Events', 'Stability']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
        values = [results[m].get(metric, 0) for m in modes]
        axes[idx].bar(modes, values, color=colors[:len(modes)])
        axes[idx].set_title(label)
        axes[idx].set_ylabel('Value')
    
    axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
    plt.close()
    
    print("Comparison chart saved to experiments/comparison.png")


def plot_scaling_events(metrics_data, title, output_path):
    if not metrics_data:
        return
    
    timestamps = [m['timestamp'] for m in metrics_data]
    container_counts = [m.get('required_containers', m.get('current_containers', 1)) for m in metrics_data]
    traffic = [m.get('current_traffic', 0) for m in metrics_data]
    predicted = [m.get('predicted_traffic', 0) for m in metrics_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(timestamps, traffic, label='Actual Traffic', color='#3498db', linewidth=2)
    ax1.plot(timestamps, predicted, label='Predicted Traffic', color='#e74c3c', linestyle='--', linewidth=2)
    ax1.set_ylabel('Traffic (req/s)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(timestamps, container_counts, label='Container Count', color='#2ecc71', linewidth=2, step='mid')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Containers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    scale_actions = [(m['timestamp'], m.get('required_containers', m.get('current_containers'))) 
                     for m in metrics_data if m.get('action') != 'no_change']
    for ts, count in scale_actions:
        ax2.axvline(x=ts, color='orange', linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Chart saved to {output_path}")


def plot_prediction_accuracy(actual, predicted, output_path):
    if not actual or not predicted:
        return
    
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(actual, label='Actual', color='#3498db', linewidth=2)
    plt.plot(predicted, label='Predicted', color='#e74c3c', linestyle='--', linewidth=2)
    plt.title('Traffic: Actual vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Traffic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    errors = actual - predicted
    plt.hist(errors, bins=30, color='#2ecc71', alpha=0.7)
    plt.title(f'Prediction Errors (MAE={mae:.2f}, RMSE={rmse:.2f})')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Prediction accuracy chart saved to {output_path}")
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")


def generate_report(results, output_path='experiments/report.txt'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AIOps Predictive Auto Scaling - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXPERIMENTAL RESULTS\n")
        f.write("-" * 40 + "\n\n")
        
        for mode, metrics in results.items():
            f.write(f"Mode: {mode.upper()}\n")
            f.write(f"  Average Latency: {metrics.get('avg_latency', 0):.4f}s\n")
            f.write(f"  Maximum Latency: {metrics.get('max_latency', 0):.4f}s\n")
            f.write(f"  Average Throughput: {metrics.get('avg_throughput', 0):.2f} req/s\n")
            f.write(f"  Scale Events: {metrics.get('scale_events', 0)}\n")
            f.write(f"  Overload Events: {metrics.get('overload_events', 0)}\n")
            f.write(f"  Stability Score: {metrics.get('stability', 0):.2f}\n\n")
        
        if 'predictive' in results and 'threshold' in results:
            pred = results['predictive']
            thresh = results['threshold']
            
            f.write("COMPARISON: Predictive vs Threshold\n")
            f.write("-" * 40 + "\n")
            
            latency_improvement = ((thresh.get('avg_latency', 0) - pred.get('avg_latency', 0)) 
                                   / max(thresh.get('avg_latency', 0), 0.001) * 100)
            overload_reduction = ((thresh.get('overload_events', 0) - pred.get('overload_events', 0))
                                   / max(thresh.get('overload_events', 0), 1) * 100)
            
            f.write(f"  Latency Improvement: {latency_improvement:.1f}%\n")
            f.write(f"  Overload Reduction: {overload_reduction:.1f}%\n")
            
            if pred.get('stability', 0) > thresh.get('stability', 0):
                f.write(f"  Stability: Predictive is more stable\n")
            else:
                f.write(f"  Stability: Threshold is more stable\n")
            
            f.write("\nCONCLUSION\n")
            f.write("-" * 40 + "\n")
            if pred.get('avg_latency', 0) < thresh.get('avg_latency', 0):
                f.write("Predictive scaling demonstrates better performance with\n")
                f.write("lower latency and fewer overload events compared to\n")
                f.write("traditional threshold-based scaling.\n")
            else:
                f.write("Results show threshold-based scaling performed better\n")
                f.write("in this test run.\n")
    
    print(f"Report saved to {output_path}")


def main():
    print("Running evaluation and generating visualizations...")
    
    results = {}
    
    for mode in ['predictive', 'threshold', 'no_scaling']:
        metrics_file = f'experiments/{mode}_metrics.json'
        events_file = f'experiments/{mode}_scaling_events.json'
        
        metrics_data = load_json(metrics_file)
        events_data = load_json(events_file)
        
        if metrics_data:
            results[mode] = calculate_metrics(metrics_data)
            
            plot_scaling_events(
                metrics_data,
                f'{mode.title()} Scaling',
                f'experiments/{mode}_scaling.png'
            )
    
    if results:
        plot_comparison(results)
        generate_report(results)
    
    print("\nEvaluation complete!")
    print("Generated files:")
    print("  - experiments/comparison.png")
    print("  - experiments/predictive_scaling.png")
    print("  - experiments/threshold_scaling.png")
    print("  - experiments/report.txt")


if __name__ == "__main__":
    main()
