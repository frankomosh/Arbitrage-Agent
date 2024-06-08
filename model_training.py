
import os
import time
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dotenv import load_dotenv
import requests
from io import StringIO

load_dotenv()

# Load environment variables
LIVE_DATA_URL = os.getenv("LIVE_DATA_URL")
MODEL_OUTPUT_PATH = os.getenv("MODEL_OUTPUT_PATH", "models/arbitrage_model.xgb")
UPDATE_INTERVAL = int(os.getenv("UPDATE_INTERVAL", 3600))  # Update every hour

# Function to fetch live data
def fetch_live_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = pd.read_csv(StringIO(response.text))
        return data
    else:
        print("Failed to fetch live data")
        return None

# Preprocess data
def preprocess_data(data):
    data = data.dropna()
    features = data.drop(columns=["profit"])
    labels = data["profit"] > 0  # Binary classification: 1 if profit > 0, else 0
    return features, labels

# Train the model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.3,
        "eval_metric": "logloss"
    }
    model = xgb.train(params, dtrain, num_boost_round=100)
    y_pred = model.predict(dtest)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy * 100.0}%")
    return model

# Save the model
def save_model(model, output_path):
    model.save_model(output_path)
    print(f"Model saved to {output_path}")

# Main function for continuous model updating
def main():
    while True:
        data = fetch_live_data(LIVE_DATA_URL)
        if data is not None:
            features, labels = preprocess_data(data)
            model = train_model(features, labels)
            save_model(model, MODEL_OUTPUT_PATH)
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
