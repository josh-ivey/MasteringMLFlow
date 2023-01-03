'''
This program creates two runs with different learning rates, 
trains and evaluates a model with those learning rates, 
and logs the evaluation results and the trained model as artifacts. 
The results of the two runs can be compared in the MLFlow UI.
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
mlflow.set_experiment("comparison-experiment")
# Start a run with a different learning rate
with mlflow.start_run(run_name="learning_rate=0.01"):
  # Define the model
  model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])

  # Compile the model with a learning rate of 0.01
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  # Train the model
  model.fit(X_train, y_train, epochs=5)

  # Evaluate the model on the test set
  eval_results = model.evaluate(X_test, y_test)

  # Log the evaluation results
  for metric_name, metric_value in zip(model.metrics_names, eval_results):
    mlflow.log_metric(metric_name, metric_value)

  # Save the model as an artifact
  mlflow.tensorflow.log_model(model, "model")

# Start a run with a different learning rate
with mlflow.start_run(run_name="learning_rate=0.001"):
  # Define the model
  model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
  ])

  # Compile the model with a learning rate of 0.001
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
  # Train the model
  model.fit(X_train, y_train, epochs=5)

  # Evaluate the model on the test set
  eval_results = model.evaluate(X_test, y_test)

  # Log the evaluation results
  for metric_name, metric_value in zip(model.metrics_names, eval_results):
    mlflow.log_metric(metric_name, metric_value)

  # Save the model as an artifact
  mlflow.tensorflow.log_model(model, "model")
