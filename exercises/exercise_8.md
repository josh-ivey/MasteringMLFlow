8. Serving a model with Flask: 

- Use the mlflow.pyfunc module to log and deploy a custom model using MLFlow, and 
then use Flask to serve the model for online predictions. 
- You should use the mlflow.pyfunc.load_model function to load the deployed 
model in Flask and create an endpoint that allows clients to send requests and 
receive predictions from the model.