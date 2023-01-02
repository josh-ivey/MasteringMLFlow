# Test the deployed model using the requests library
import requests

response = requests.post("http://localhost:8080/invocations", json={
    "input_1": "test",
    "input_2": 123
})
prediction = response.json()
print(prediction)
