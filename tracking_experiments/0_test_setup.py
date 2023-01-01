'''In the first file, you've demonstrated how to set up an experiment 
and log a parameter and metric. This is a good start for introducing 
experiment tracking with MLFlow.'''

import mlflow

mlflow.set_experiment('test-experiment')

with mlflow.start_run():
    mlflow.log_param("some_param", 23.33) 
    mlflow.log_metric("some_metric", 1.0)
