import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from xgboost import XGBClassifier

CSV_FILE = "training_data.csv"
MODEL_FILE = "ml_model.pkl"

def train_model():
    """Train an XGBoost model to optimize system performance."""
    df = pd.read_csv(CSV_FILE)

    # Split features (X) and labels (y)
    X = df[["cpu_usage", "ram_usage", "disk_usage", "gpu_usage"]]
    y = df["action"]

    # Convert category labels to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Save label encoder for decoding predictions later
    joblib.dump(label_encoder, "label_encoder.pkl")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(eval_metric="mlogloss")
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_FILE)
    print(f"XGBoost Model trained and saved as {MODEL_FILE}")

if __name__ == "__main__":
    train_model()