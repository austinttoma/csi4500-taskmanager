import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

CSV_FILE = "training_data.csv"
MODEL_FILE = "ml_model.pkl"

def train_model():
    """Train an XGBoost model to predict a numeric priority based on runtime."""
    df = pd.read_csv(CSV_FILE)
    
    # We assume your CSV now has "runtime_seconds" and "numeric_priority" columns.
    # runtime_seconds: How long the process has been running (in seconds)
    # numeric_priority: A number (1-10) where higher means more important
    X = df[["runtime_seconds"]]
    y = df["numeric_priority"]

    # Split the data into training and test sets (optional: can also train on full data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train an XGBoost Regressor
    model = XGBRegressor(eval_metric="rmse")
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, MODEL_FILE)
    print(f"XGBoost Regressor trained and saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()

