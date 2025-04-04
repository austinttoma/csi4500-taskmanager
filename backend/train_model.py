import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

CSV_FILE = "training_data.csv"
MODEL_FILE = "ml_model.pkl"

def train_model():
    """Train an XGBoost model to predict a numeric priority based on process runtime.
    
    Expects a CSV file with columns:
        - runtime_seconds: How long the process has been running (in seconds)
        - numeric_priority: A numeric value (1-10) representing process priority.
    Saves the trained model to MODEL_FILE.
    """
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run collect_training_data.py first.")
        return

    df = pd.read_csv(CSV_FILE)
    
    # Ensure the required columns exist
    if "runtime_seconds" not in df.columns or "numeric_priority" not in df.columns:
        print("Error: CSV file does not have the expected columns 'runtime_seconds' and 'numeric_priority'.")
        return

    X = df[["runtime_seconds"]]
    y = df["numeric_priority"]

    # Optionally split the data (here we train on 80% of the data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost Regressor with RMSE as evaluation metric
    model = XGBRegressor(eval_metric="rmse")
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"XGBoost Regressor trained and saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()


