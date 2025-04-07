import sys
import os
import psutil
import subprocess
import time
import joblib
import difflib
import numpy as np
from functools import partial
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QComboBox, QMessageBox, QLineEdit, QHeaderView
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon

# Load pre-trained ML model
MODEL_PATH = "ml_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

from optimizer import clean_memory  # <-- Using optimizer module

# Optionally import the training script for in-app retraining
try:
    import train_model
except ImportError:
    train_model = None

class MonitoringThread(QThread):
    metrics_signal = pyqtSignal(dict)

    def run(self):
        while True:
            metrics = {
                "cpu_usage": psutil.cpu_percent(interval=1),
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
    except Exception as e:
        print(f"GPU usage retrieval failed: {e}")
        return 0.0  # No GPU detected or error

def get_process_list():
    """
    Returns an aggregated list of processes grouped by name.
    For each unique process name, computes:
      - count: number of processes with that name
      - avg runtime (sec)
      - predicted priority (using the average runtime)
      - list of process IDs (pids)
    """
    global model
    now = time.time()
    groups = {}
    for proc in psutil.process_iter(['pid', 'name', 'create_time']):
        try:
            runtime = now - proc.info['create_time']
            name = proc.info['name']
            pid = proc.info['pid']
            if name not in groups:
                groups[name] = {"name": name, "pids": [], "runtimes": []}
            groups[name]["pids"].append(pid)
            groups[name]["runtimes"].append(runtime)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    aggregated_list = []
    for name, data in groups.items():
        count = len(data["pids"])
        avg_runtime = sum(data["runtimes"]) / count if count > 0 else 0
        if model is not None:
            pred = model.predict([[avg_runtime]])[0]
        else:
            pred = 0
        aggregated_list.append({
            "name": name,
            "pids": data["pids"],
            "count": count,
            "runtime": avg_runtime,
            "priority": pred
        })
    return aggregated_list

class SystemOptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        self.monitor_thread = MonitoringThread()
        self.monitor_thread.metrics_signal.connect(self.update_status)
        self.monitor_thread.start()

        # Auto-refresh process table every 10 seconds
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.update_process_table)
        self.refresh_timer.start(10000)
    
    # Setting a dark mode 
        dark_stylesheet = """
        QWidget {
                background-color: #121212;
                color: #ffffff;
        }       

        QTableWidget {
                background-color: #1e1e1e;
                gridline-color: #444;
                color: #ffffff;
        }

        QHeaderView::section {
                background-color: #2c2c2c;
                color: #ffffff;
                padding: 4px;
                border: 1px solid #444;
        }

        QTableWidget QTableCornerButton::section {
                background-color: #2c2c2c;
        }

        QScrollBar:vertical, QScrollBar:horizontal {
                background: #2c2c2c;
                border: none;
        }
        """
        self.setStyleSheet(dark_stylesheet)
        self.setWindowIcon(QIcon("icon.png"))

    def initUI(self):
        self.setWindowTitle("System Optimizer")
        self.setGeometry(100, 100, 900, 600)
        

        main_layout = QVBoxLayout()

        # Top layout for status, search bar, and sort dropdown
        top_layout = QHBoxLayout()
        self.status_label = QLabel("System Status: Monitoring...", self)
        top_layout.addWidget(self.status_label)

        # Search bar
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search process name...")
        self.search_bar.textChanged.connect(self.update_process_table)
        top_layout.addWidget(self.search_bar)

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
        

        # Buttons panel
        btn_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Process List", self)
        self.refresh_button.clicked.connect(self.update_process_table)
        btn_layout.addWidget(self.refresh_button)

        self.optimize_button = QPushButton("Optimize RAM", self)
        self.optimize_button.clicked.connect(self.optimize_ram)
        btn_layout.addWidget(self.optimize_button)

        self.retrain_button = QPushButton("Retrain Model", self)
        self.retrain_button.clicked.connect(self.retrain_model)
        btn_layout.addWidget(self.retrain_button)

        self.ml_button = QPushButton("ML Suggestions", self)
        self.ml_button.clicked.connect(lambda: self.ml_suggestions_loop())
        btn_layout.addWidget(self.ml_button)

        main_layout.addLayout(btn_layout)

        # Table to display aggregated process data:
        # Columns: Count, Process Name, Avg Runtime (sec), Priority, CPU (%), Memory (MB), Action
        self.table = QTableWidget(self)
        self.table.setSortingEnabled(True)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Count", "Process Name", "Avg Runtime (sec)", "Priority",
            "CPU (%)", "Memory (MB)", "Action"
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)

        main_layout.addWidget(self.table)

        self.setLayout(main_layout)
        QTimer.singleShot(500, self.update_process_table)

    def update_status(self, metrics):
        """Update UI with real-time system metrics."""
        self.status_label.setText(
            f"CPU: {metrics['cpu_usage']}% | RAM: {metrics['ram_usage']}% | "
            f"Disk: {metrics['disk_usage']}% | GPU: {metrics['gpu_usage']}%"
        )

    def update_process_table(self):
        processes = get_process_list()
        search_term = self.search_bar.text().lower().strip()
        if search_term:
            # Calculate a similarity score for each process based on the search term
            for p in processes:
                p["similarity"] = difflib.SequenceMatcher(None, search_term, p["name"].lower()).ratio()
            # Sort processes so that those with higher similarity come first
            processes.sort(key=lambda x: x["similarity"], reverse=True)
        else:
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
        for row, group in enumerate(processes):
            self.create_process_row(row, group)

    def create_process_row(self, row, group):
        """Helper to create a table row for an aggregated process group."""
        self.table.setItem(row, 0, QTableWidgetItem(str(group["count"])))
        self.table.setItem(row, 1, QTableWidgetItem(str(group["name"])))
        self.table.setItem(row, 2, QTableWidgetItem(f"{group['runtime']:.1f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{group['priority']:.2f}"))

        # Aggregate CPU and memory usage for all processes in the group
        total_cpu = 0.0
        total_mem = 0.0
        for pid in group["pids"]:
            try:
                proc = psutil.Process(pid)
                total_cpu += proc.cpu_percent(interval=0)
                total_mem += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        self.table.setItem(row, 4, QTableWidgetItem(f"{total_cpu:.1f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{total_mem:.1f}"))

        # Create a "Close" button for each row using partial to bind the group's pids
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(partial(self.close_process_group, group["pids"]))
        self.table.setCellWidget(row, 6, close_button)

    def close_process_group(self, pids):
        """Terminate all processes in the given group and refresh the table."""
        errors = []
        for pid in pids:
            try:
                psutil.Process(pid).terminate()
            except Exception as e:
                errors.append(f"{pid}: {e}")
        if errors:
            self.status_label.setText("Errors closing some processes: " + ", ".join(errors))
        else:
            self.status_label.setText("Closed all processes in the group.")
        self.update_process_table()

    def optimize_ram(self):
        clean_memory()
        self.status_label.setText("Ran RAM optimization.")
        self.update_process_table()

    def retrain_model(self):
        """Retrain the model and reload it."""
        global model
        if train_model is None:
            self.status_label.setText("Retrain model functionality not available.")
            return
        try:
            train_model.train_model()
            model = joblib.load(MODEL_PATH)
            self.status_label.setText("Model retrained and reloaded successfully.")
            self.update_process_table()
        except Exception as e:
            self.status_label.setText(f"Model retrain failed: {e}")

    def get_ml_suggestions(self):
        """
        Build a list of candidate process groups for ML suggestion.
        Each candidate is one with predicted priority <= 3.
        The suggestion includes aggregated CPU and memory usage as a score.
        """
        suggestions = []
        groups = get_process_list()
        for group in groups:
            if group["priority"] <= 4:
                total_cpu = 0.0
                total_mem = 0.0
                for pid in group["pids"]:
                    try:
                        proc = psutil.Process(pid)
                        total_cpu += proc.cpu_percent(interval=0)
                        total_mem += proc.memory_info().rss / (1024 * 1024)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                score = total_cpu + total_mem
                suggestion = {
                    "name": group["name"],
                    "count": group["count"],
                    "pids": group["pids"],
                    "total_cpu": total_cpu,
                    "total_mem": total_mem,
                    "priority": group["priority"],
                    "score": score
                }
                suggestions.append(suggestion)
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions

    def ml_suggestions_loop(self, rejected=None):
        if rejected is None:
            rejected = []

        # Check if current RAM usage is under 65% (stopping condition)
        if psutil.virtual_memory().percent < 65:
            QMessageBox.information(
                self,
                "System Optimized",
                "RAM usage is now under 65%. System optimized!"
            )
            return

        suggestions = [s for s in self.get_ml_suggestions() if s["name"] not in rejected]

        if not suggestions:
            cont = QMessageBox.question(
                self,
                "System Optimized",
                "No further suggestions available. Would you like to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if cont == QMessageBox.Yes:
                self.status_label.setText("System remains optimized.")
            else:
                self.status_label.setText("Exiting ML suggestion loop.")
            return

        # Take the highest-scoring suggestion
        suggestion = suggestions[0]
        msg = (f"ML Suggestion:\nClose '{suggestion['name']}' "
               f"({suggestion['count']} instances)\n"
               f"Aggregated CPU: {suggestion['total_cpu']:.1f}% | "
               f"Memory: {suggestion['total_mem']:.1f} MB\n"
               f"Justification: Low priority ({suggestion['priority']:.2f}) "
               f"and high resource usage (Score: {suggestion['score']:.1f}).\n\n"
               "Do you want to accept this suggestion?")
    
        # Create a custom message box with three buttons
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("ML Suggestion")
        msgBox.setText(msg)
        acceptButton = msgBox.addButton("Accept", QMessageBox.AcceptRole)
        rejectButton = msgBox.addButton("Reject", QMessageBox.RejectRole)
        exitButton = msgBox.addButton("Exit", QMessageBox.DestructiveRole)
        msgBox.exec_()

        clicked = msgBox.clickedButton()
        if clicked == acceptButton:
            # Accept suggestion: close processes and restart loop.
            self.close_process_group(suggestion["pids"])
            self.ml_suggestions_loop()
        elif clicked == rejectButton:
            # Reject suggestion: add to rejected list and continue.
            rejected.append(suggestion["name"])
            self.ml_suggestions_loop(rejected)
        elif clicked == exitButton:
            # Exit the loop.
            self.status_label.setText("Exiting ML suggestion loop.")
            return

    def ml_suggestion(self):
        """Kick off the ML suggestion loop."""
        self.ml_suggestions_loop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SystemOptimizerApp()
    window.show()
    sys.exit(app.exec_())
