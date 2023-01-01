import requests
import json

# Set the endpoint URL of the model
endpoint_url = "http://localhost:8000/predict"

# Set the input data
X = [[5.1, 3.5, 1.4, 0.2], [5.9, 3.0, 5.1, 1.8]]

# Send a request to the endpoint with the input data
response = requests.post(endpoint_url, json={"X": X})

# Print the predictions
predictions = response.json()
print(predictions)
