import mlflow
import json

# Load the model from a file
model = mlflow.sklearn.load_model("models/random_forest")

# Define the input and output schema for the model
input_schema = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "number"
            }
        }
    }
}
output_schema = {
    "type": "object",
    "properties": {
        "prediction": {"type": "number"}
    }
}

# Serve the model as a REST API
mlflow.pyfunc.serve(model=model, port=8080, input_schema=input_schema, output_schema=output_schema)

