import psutil
import subprocess
import csv
import random
import time

CSV_FILE = "training_data.csv"

def get_gpu_usage():
    """Get GPU usage if available (for NVIDIA GPUs)."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        return float(output.strip())
    except Exception:
        return 0.0  # No GPU detected or error

def collect_sample_data():
    """Collect system metrics and label an optimization action manually."""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    disk_usage = psutil.disk_usage('/').percent
    gpu_usage = get_gpu_usage()

    # Label the best optimization strategy based on conditions
    if cpu_usage > 80:
        action = "lower_cpu_priority"
    elif ram_usage > 85:
        action = "clear_cache"
    elif gpu_usage > 90:
        action = "limit_gpu_usage"
    else:
        action = "no_action"

    return [cpu_usage, ram_usage, disk_usage, gpu_usage, action]

def save_data():
    """Generate and save training data."""
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cpu_usage", "ram_usage", "disk_usage", "gpu_usage", "action"])  # Header

        for _ in range(500):  # Collect 500 samples
            row = collect_sample_data()
            writer.writerow(row)
            time.sleep(0.5)  # Delay to capture real-world variations

    print(f"✅ Training data saved to {CSV_FILE}")

if __name__ == "__main__":
    save_data()
