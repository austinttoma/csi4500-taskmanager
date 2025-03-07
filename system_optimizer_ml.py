import sys
import os
import psutil
import subprocess
import time
import joblib
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import QThread, pyqtSignal

# Load pre-trained ML model (to be trained separately)
MODEL_PATH = "ml_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    
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

# Load the label encoder for decoding predictions
LABEL_ENCODER_PATH = "label_encoder.pkl"
if os.path.exists(LABEL_ENCODER_PATH):
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
else:
    label_encoder = None

def optimize_system():
    """Optimize system performance using ML predictions."""
    global model, label_encoder
    if model is None or label_encoder is None:
        return "No ML model or label encoder found. Train and save both first."
    
    metrics = np.array([
        [
            psutil.cpu_percent(),
            psutil.virtual_memory().percent,
            psutil.disk_usage('/').percent,
            get_gpu_usage()
        ]
    ])
    
    # Predict numerical action and decode it
    predicted_action_numeric = model.predict(metrics)[0]
    action = label_encoder.inverse_transform([predicted_action_numeric])[0]

    # Perform system optimization based on decoded action
    if action == "lower_cpu_priority":
        subprocess.run(["renice", "+10", "-p", str(os.getpid())])
        return "Lowered CPU priority."
    elif action == "clear_cache":
        subprocess.run(["sync; echo 3 | sudo tee /proc/sys/vm/drop_caches"], shell=True)
        return "Cleared memory cache."
    elif action == "limit_gpu_usage":
        subprocess.run(["nvidia-smi", "-pl", "100"], shell=True)
        return "Limited GPU power usage."
    else:
        return "No optimization needed."

class SystemOptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.monitor_thread = MonitoringThread()
        self.monitor_thread.metrics_signal.connect(self.update_status)
        self.monitor_thread.start()
    
    def initUI(self):
        self.setWindowTitle("System Optimizer")
        self.setGeometry(100, 100, 300, 200)
        
        layout = QVBoxLayout()
        self.status_label = QLabel("System Status: Monitoring...", self)
        layout.addWidget(self.status_label)
        
        self.optimize_button = QPushButton("Start Optimization", self)
        self.optimize_button.clicked.connect(self.run_optimization)
        layout.addWidget(self.optimize_button)
        
        self.setLayout(layout)
        
    def update_status(self, metrics):
        """Update UI with real-time system metrics."""
        self.status_label.setText(
            f"CPU: {metrics['cpu_usage']}% | RAM: {metrics['ram_usage']}% | "
            f"Disk: {metrics['disk_usage']}% | GPU: {metrics['gpu_usage']}%"
        )
    
    def run_optimization(self):
        result = optimize_system()
        self.status_label.setText(f"System Status: {result}")
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SystemOptimizerApp()
    window.show()
    sys.exit(app.exec_())
