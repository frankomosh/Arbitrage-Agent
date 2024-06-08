# import datetime

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import yfinance as yf
# from sklearn.metrics import mean_squared_error as mse


# def download_data():
#     uni_ticker = "UNI-USD"
#     eth_ticker = "ETH-USD"
#     start = datetime.datetime(2019, 1, 1)
#     end = datetime.datetime(2024, 4, 1)
#     uni = yf.download(uni_ticker, start=start, end=end, interval="1d")
#     eth = yf.download(eth_ticker, start=start, end=end, interval="1d")
#     uni = uni.reset_index()
#     uni.to_csv("uni.csv", index=False)
#     eth = eth.reset_index()
#     eth.to_csv("eth.csv", index=False)
#     return uni, eth


# def process_data(uni: pd.DataFrame, eth: pd.DataFrame):
#     uni = uni[uni["Open"] < 0.30]
#     uni = uni[["Date", "Open"]]
#     eth = eth[["Date", "Open"]]

#     uni.rename(columns={"Open": "UNI"}, inplace=True)
#     eth.rename(columns={"Open": "ETH"}, inplace=True)

#     df = pd.merge(uni, eth, on="Date")
#     df.dropna(inplace=True)
#     df["price"] = df["ETH"] / df["UNI"]
#     ret = 100 * (df["price"].pct_change()[1:])
#     realized_vol = ret.rolling(5).std()
#     realized_vol = pd.DataFrame(realized_vol)
#     realized_vol.reset_index(drop=True, inplace=True)
#     returns_svm = ret**2  # returns squared
#     returns_svm = returns_svm.reset_index()
#     X = pd.concat([realized_vol, returns_svm], axis=1, ignore_index=True)
#     X = X[4:].copy()
#     X = X.reset_index()
#     X.drop("index", axis=1, inplace=True)
#     X.drop(1, axis=1, inplace=True)
#     X.rename(columns={0: "realized_vol", 2: "returns_squared"}, inplace=True)
#     X["target"] = X["realized_vol"].shift(-5)
#     X.dropna(inplace=True)
#     Y = X["target"]
#     X.drop("target", axis=1, inplace=True)
#     n = 252
#     X_train = X.iloc[:-n]
#     X_test = X.iloc[-n:]
#     Y_train = Y.iloc[:-n]
#     Y_test = Y.iloc[-n:]
#     return X_train, X_test, Y_train, Y_test


# def train_model(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     Y_train: pd.DataFrame,
#     Y_test: pd.DataFrame,
# ):
#     model = nn.Sequential(
#         nn.Linear(X_train.shape[1], 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.ReLU(),
#         nn.Linear(64, 1),
#     )

#     # Loss and optimizer
#     criterion = nn.MSELoss()
#     optimizer = optim.RMSprop(model.parameters())

#     # Convert data to PyTorch tensors
#     X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
#     y_tensor = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

#     # Training loop
#     epochs_trial = np.arange(100, 400, 4)
#     batch_trial = np.arange(100, 400, 4)
#     DL_pred = []
#     DL_RMSE = []

#     for i, j, k in zip(range(4), epochs_trial, batch_trial):
#         for epoch in range(j):
#             optimizer.zero_grad()
#             outputs = model(X_tensor)
#             loss = criterion(outputs, y_tensor)
#             loss.backward()
#             optimizer.step()

#         with torch.no_grad():
#             DL_predict = model(X_test_tensor).numpy()
#             DL_RMSE.append(
#                 np.sqrt(mse(Y_test.values / 100, DL_predict.flatten() / 100))
#             )
#             DL_pred.append(DL_predict)
#             print("DL_RMSE_{}:{:.6f}".format(i + 1, DL_RMSE[i]))

#     return model


# def serialize_to_onnx(
#     model: nn.Module, X_train: pd.DataFrame, save_path="torch_vol_model"
# ):
#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Dummy input matching the input size
#     sample_input = torch.randn(
#         1, X_train.shape[1]
#     )  # Replace 1 with the batch size you'd like to use

#     # Specify the path to save the ONNX model
#     onnx_file_path = save_path + ".onnx"

#     torch.onnx.export(
#         model,
#         sample_input,
#         onnx_file_path,
#         export_params=True,
#         opset_version=10,
#         do_constant_folding=True,
#         input_names=["input"],
#         output_names=["output"],
#         dynamic_axes={
#             "input": {0: "batch_size"},
#             "output": {0: "batch_size"},
#         },
#     )
#     print(f"Saved serialized ONNX model to {onnx_file_path}.")


# def main():
#     uni, eth = download_data()
#     X_train, X_test, Y_train, Y_test = process_data(uni, eth)
#     model = train_model(X_train, X_test, Y_train, Y_test)
#     serialize_to_onnx(model, X_train)


# if __name__ == "__main__":
#     main()

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
