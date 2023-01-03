import mlflow
import mlflow.sklearn
import flask
import json

# Load the saved model
model = mlflow.sklearn.load_model("models/random_forest")

# Create a `flask` server to serve the model
app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Extract the input data from the request
    request_data = flask.request.data.decode("utf-8")
    data = json.loads(request_data)
    X = data["X"]
    
    # Use the model to make predictions
    predictions = model.predict(X).tolist()
    
    # Return the predictions in the response
    return flask.Response(response=json.dumps(predictions), status=200, mimetype="application/json")

# Run the server
app.run(host='0.0.0.0', port=8000)
