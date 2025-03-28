import psutil
import csv
import time
import numpy as np

CSV_FILE = "training_data.csv"

def collect_usage(duration=60, interval=5):
    """
    Tracks running processes over a period of 'duration' seconds,
    sampling every 'interval' seconds. Returns a dictionary mapping
    process names to their current runtime (in seconds).
    """
    usage = {}
    start_time = time.time()
    while time.time() - start_time < duration:
        now = time.time()
        for proc in psutil.process_iter(['name', 'create_time']):
            try:
                name = proc.info['name']
                # Calculate how long the process has been running
                runtime = now - proc.info['create_time']
                usage[name] = runtime
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        time.sleep(interval)
    return usage

def assign_priority_numeric(usage_data):
    """
    Assigns a numeric priority based on the runtime percentiles.
    Higher runtime means a higher priority (i.e., a process you want to keep).
    For instance, processes in the top 10% (90th-100th percentile) get 10,
    those in the next decile get 9, and so on.
    Returns a dictionary mapping process name to a priority number (1-10).
    """
    runtimes = np.array(list(usage_data.values()))
    if len(runtimes) == 0:
        return {}
    
    # Compute decile thresholds: 10%, 20%, ..., 90%
    thresholds = [np.percentile(runtimes, p) for p in range(10, 100, 10)]
    
    priority_dict = {}
    for proc, runtime in usage_data.items():
        # Default lowest priority if below the lowest threshold
        priority = 1  
        # Check which decile the runtime falls into
        for i, thresh in enumerate(thresholds):
            if runtime >= thresh:
                priority = i + 2  # i starts at 0, so add 2
        # Cap priority at 10
        if priority > 10:
            priority = 10
        priority_dict[proc] = priority
    return priority_dict

def save_training_data(usage_data, priority_data):
    """
    Writes the collected usage data to a CSV file with columns:
    process_name, runtime_seconds, and numeric_priority.
    """
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["process_name", "runtime_seconds", "numeric_priority"])
        for proc_name, runtime in usage_data.items():
            priority = priority_data.get(proc_name, 0)
            writer.writerow([proc_name, round(runtime, 1), priority])
    print(f"✅ Training data saved to {CSV_FILE}")

def main():
    # Track processes for 60 seconds, sampling every 5 seconds
    duration = 60
    interval = 5
    usage_data = collect_usage(duration=duration, interval=interval)
    priority_data = assign_priority_numeric(usage_data)
    save_training_data(usage_data, priority_data)

if __name__ == "__main__":
    main()

