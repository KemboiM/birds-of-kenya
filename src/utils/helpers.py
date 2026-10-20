import json
import os
import shutil
from datetime import datetime


def save_label_list(class_names, config_dir, project, version):
    """Write the label list (one species per line, line N = class index N).

    Saves a versioned copy (birds_label_list_1.0.txt) and refreshes
    birds_label_list.txt so it always matches the latest model.
    """
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, f"{project}_label_list_{version}.txt")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(class_names) + "\n")
    shutil.copy2(path, os.path.join(config_dir, f"{project}_label_list.txt"))
    print(f"  Label list saved to {path} ({len(class_names)} species)")
    return path


def load_label_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def log_training(run_name, epochs, batch_size, metrics, log_file="logs/training_log.json"):
    """Append one entry per training run."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'run': run_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'metrics': metrics,
    }
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logs = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    print(f"Training logged to {log_file}")
