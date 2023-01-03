import mlflow.projects

# Set the URI of the project to run
project_uri = "file://."  # Replace with the URI of your project

# Set the parameters for the run
params = {"data_file": "data/iris.csv", "regularization": 0.1}

# Run the project and create a reproducible environment on a local host
mlflow.projects.run(project_uri, parameters=params, env_manager="local")

