'''
In this file, mlflow is being used to track the training and evaluation 
of a convolutional neural network model for the MNIST dataset. The mlflow 
library is used to create an experiment named "tf-mnist-experiment", and a 
new run is started within this experiment using the mlflow.start_run() function. 
Within this run, the model is defined, compiled, and trained using the 
TensorFlow Keras API. After training, the model is evaluated on the 
test set and the evaluation metrics (loss and accuracy) are logged using the 
mlflow.log_metric() function. 
Finally, the trained model is saved as an artifact using the 
mlflow.tensorflow.log_model() function. This allows the model to be saved 
and later retrieved within the mlflow UI or using the mlflow library.
'''
import mlflow
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train[..., tf.newaxis]
X_test = X_test[..., tf.newaxis]

# Create an experiment
mlflow.set_experiment("tf-mnist-experiment")

# Start a run
with mlflow.start_run():
  # Define the model
  model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])

  # Compile the model
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  # Train the model
  model.fit(X_train, y_train, epochs=5)

  # Evaluate the model on the test set
  eval_results = model.evaluate(X_test, y_test)

  # Log the evaluation results
  for metric_name, metric_value in zip(model.metrics_names, eval_results):
    mlflow.log_metric(metric_name, metric_value)

  # Save the model as an artifact
  mlflow.tensorflow.log_model(model=model, artifact_path="model")
