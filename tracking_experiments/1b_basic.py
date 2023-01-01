'''The second and third files show how to log model artifacts, 
which is an important aspect of experiment tracking as it allows 
you to save and retrieve the models that you've trained. 
It's also good that you've demonstrated two different ways to 
log model artifacts (with mlflow.sklearn.log_model and by saving 
the model object as a pickle file and logging it as an artifact).'''

import mlflow
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
X, y = load_iris(return_X_y=True)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an experiment
mlflow.set_experiment("iris-experiment-2")
# Start a run
with mlflow.start_run():
  # Train a random forest classifier
  model = RandomForestClassifier()
  model.fit(X_train, y_train)
  # Evaluate the model on the test set
  accuracy = model.score(X_test, y_test)

  # Log the accuracy metric
  mlflow.log_metric("accuracy", accuracy)

  # Save the model as an artifact
  with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
  mlflow.log_artifact("model.pkl")
