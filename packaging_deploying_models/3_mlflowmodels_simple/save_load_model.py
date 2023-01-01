'''
This file demonstrates how to use MLFlow Models to save and load machine learning models, 
and how to use these saved models to make predictions on new data. Using MLFlow Models can
be particularly useful in cases where you want to save and load models in a variety of 
formats (e.g. scikit-learn, PyTorch, TensorFlow) and serialization libraries 
(e.g. pickle, joblib). It can also be useful for managing the versioning and deployment of 
models, as it allows you to easily save and load models to and from a variety of locations, 
including local files and cloud storage systems like AWS S3.
'''
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split the data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a local file
mlflow.sklearn.save_model(model, "models/random_forest")

# Save the model to S3 (uncomment to run)
# mlflow.sklearn.save_model(model, "s3://my-bucket/models/random_forest")

# Load the model from a local file
loaded_model = mlflow.sklearn.load_model("models/random_forest")

# Load the model from S3 (uncomment to run)
# loaded_model = mlflow.sklearn.load_model("s3://my-bucket/models/random_forest")

# Use the model to make predictions
predictions = loaded_model.predict(X_test)
