import sys
import os
import psutil
import subprocess
import time
import joblib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer

# Load pre-trained ML model
MODEL_PATH = "ml_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

from optimizer import clean_memory  # <-- Added import

class MonitoringThread(QThread):
    metrics_signal = pyqtSignal(dict)

    def run(self):
        while True:
            metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "ram_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "gpu_usage": get_gpu_usage()
            }
            self.metrics_signal.emit(metrics)
            time.sleep(2)

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

def get_process_list():
    """Returns a list of running processes with predicted numeric priority."""
    global model
    processes = []
    now = time.time()
    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
        try:
            runtime = now - proc.info['create_time']
            processes.append({
                "pid": proc.info['pid'],
                "name": proc.info['name'],
                "runtime": runtime
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not processes:
        return []

    # Prepare features for prediction: our model expects a 2D array with runtime_seconds
    features = np.array([[p["runtime"]] for p in processes])
    predicted_priorities = model.predict(features) if model is not None else [0] * len(processes)
    for i, p in enumerate(processes):
        p["priority"] = predicted_priorities[i]

    return processes

class SystemOptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.monitor_thread = MonitoringThread()
        self.monitor_thread.metrics_signal.connect(self.update_status)
        self.monitor_thread.start()

    def initUI(self):
        self.setWindowTitle("System Optimizer")
        self.setGeometry(100, 100, 900, 600)

        main_layout = QVBoxLayout()

        # Top layout for status and sort dropdown
        top_layout = QHBoxLayout()

        self.status_label = QLabel("System Status: Monitoring...", self)
        top_layout.addWidget(self.status_label)

        # Add sort dropdown (combo box)
        self.sort_combo = QComboBox(self)
        self.sort_combo.addItems([
            "Priority: Low to High",
            "Priority: High to Low",
            "Process Name: A-Z",
            "Process Name: Z-A"
        ])
        self.sort_combo.currentIndexChanged.connect(self.update_process_table)
        top_layout.addWidget(QLabel("Sort By:", self))
        top_layout.addWidget(self.sort_combo)

        main_layout.addLayout(top_layout)

        # Button to refresh the process list display
        self.refresh_button = QPushButton("Refresh Process List", self)
        self.refresh_button.clicked.connect(self.update_process_table)
        main_layout.addWidget(self.refresh_button)

        # Button to optimize RAM
        self.optimize_button = QPushButton("Optimize RAM", self)
        self.optimize_button.clicked.connect(self.optimize_ram)
        main_layout.addWidget(self.optimize_button)

        # Table to display processes, priorities, and a Close button for each row
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["PID", "Process Name", "Runtime (sec)", "Priority", "Action"])
        main_layout.addWidget(self.table)

        self.setLayout(main_layout)

        # Delay initial update to let GUI render
        QTimer.singleShot(500, self.update_process_table)

    def update_status(self, metrics):
        """Update UI with real-time system metrics."""
        self.status_label.setText(
            f"CPU: {metrics['cpu_usage']}% | RAM: {metrics['ram_usage']}% | "
            f"Disk: {metrics['disk_usage']}% | GPU: {metrics['gpu_usage']}%"
        )

    def update_process_table(self):
        """Update the table widget with the sorted process list based on the selected sort option."""
        processes = get_process_list()
        sort_option = self.sort_combo.currentText()

        if sort_option == "Priority: Low to High":
            processes.sort(key=lambda x: x["priority"])
        elif sort_option == "Priority: High to Low":
            processes.sort(key=lambda x: x["priority"], reverse=True)
        elif sort_option == "Process Name: A-Z":
            processes.sort(key=lambda x: x["name"].lower())
        elif sort_option == "Process Name: Z-A":
            processes.sort(key=lambda x: x["name"].lower(), reverse=True)

        self.table.setRowCount(len(processes))
        for row, p in enumerate(processes):
            self.table.setItem(row, 0, QTableWidgetItem(str(p["pid"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(p["name"])))
            self.table.setItem(row, 2, QTableWidgetItem(f"{p['runtime']:.1f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{p['priority']:.2f}"))

            # Create a "Close" button for each row
            close_button = QPushButton("Close", self)
            close_button.clicked.connect(lambda checked, pid=p["pid"]: self.close_process(pid))
            self.table.setCellWidget(row, 4, close_button)

    def close_process(self, pid):
        """Terminate the process with the given PID and refresh the table."""
        try:
            psutil.Process(pid).terminate()
            self.status_label.setText(f"Closed process with PID {pid}.")
        except Exception as e:
            self.status_label.setText(f"Failed to close process with PID {pid}: {e}")
        self.update_process_table()

    def optimize_ram(self):
        clean_memory()
        self.status_label.setText("Ran RAM optimization.")
        self.update_process_table()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SystemOptimizerApp()
    window.show()
    sys.exit(app.exec_())