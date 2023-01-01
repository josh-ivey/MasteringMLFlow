# train.py
import argparse
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, required=True)
parser.add_argument("--regularization", type=float, default=0.1)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_file)
X = data.drop("target", axis=1)
y = data["target"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(C=args.regularization)
model.fit(X_train, y_train)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Log results to MLFlow
mlflow.log_param("regularization", args.regularization)
