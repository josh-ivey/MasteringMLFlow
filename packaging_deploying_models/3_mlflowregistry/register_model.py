import mlflow

def register_model(model_name, model_path, model_version, metadata=None):
    """
    Registers a model with the specified name, version, and metadata in the MLFlow Model registry.
    
    :param model_name: The name of the model.
    :param model_path: The file path or URI of the model.
    :param model_version: The version of the model.
    :param metadata: A dictionary of metadata to associate with the model.
    """
    if metadata is None:
        metadata = {}
    mlflow.register_model(model_name=model_name, model_path=model_path, model_version=model_version, metadata=metadata)
    
    
if __name__ == "__main__":
    model_name = "random_forest"
    model_path = "models/random_forest"
    model_version = "1.0"
    metadata = {"author": "Jane Doe"}
    
    register_model(model_name, model_path, model_version, metadata)
