import mlflow

# Create an experiment
mlflow.set_experiment("custom-experiment")
# Start a new run
with mlflow.start_run():
  # Log a custom metric
  mlflow.log_metric("custom_metric1", 0.95, step=1)
  mlflow.log_metric("custom_metric2", 0.97, step=2)
  mlflow.log_metric("custom_metric3", 0.99, step=3)

  # Log a custom parameter
  mlflow.log_param("custom_parameter1", "some value")
  mlflow.log_param("custom_parameter2", "some value")

  # Log a custom artifact
  with open("custom_artifact.txt", "w") as f:
    f.write("This is a custom artifact")
  mlflow.log_artifact("custom_artifact.txt")
