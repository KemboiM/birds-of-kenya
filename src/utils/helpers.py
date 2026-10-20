import json
import os
from datetime import datetime

def save_metrics(metrics, species_name):
    """Save training metrics to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{species_name}_metrics_{timestamp}.json"
    
    os.makedirs("results", exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {filename}")

def save_report(report, run_name):
    """Save the per-species classification report to JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{run_name}_per_species_{timestamp}.json"
    os.makedirs("results", exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Per-species report saved to {filename}")


def save_class_names(class_names, path="models/class_names.json"):
    """Save label order so predictions can be mapped back to species names."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"  Class names saved to {path}")


def log_training(species_name, epochs, batch_size, metrics):
    """Log training details."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'species': species_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'metrics': metrics
    }
    
    os.makedirs("logs", exist_ok=True)
    log_file = "logs/training_log.json"
    
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    print(f"Training logged to {log_file}")
