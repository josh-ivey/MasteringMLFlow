# validate.py
import argparse
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, required=True)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_file)
X = data.drop("target", axis=1)
y = data["target"]

# Load trained model
model = pickle.load("models/model.pkl")

# Evaluate model
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Accuracy: {acc:.3f}")
mlflow.log_metric("accuracy", acc)

