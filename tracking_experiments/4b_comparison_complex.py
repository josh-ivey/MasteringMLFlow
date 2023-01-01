'''This script trains and evaluates a convolutional neural network (CNN) on the MNIST dataset 
using TensorFlow and logs the evaluation results and trained model as artifacts using MLFlow. 
The script compares the performance of the CNN when trained using three different 
learning rates, 0.01, 0.001, and 0.0001. For each learning rate, the script creates a new run 
in the "comparison-experiment" experiment, compiles the model with the corresponding learning rate, 
trains the model for 5 epochs, evaluates the model on the test set, logs the evaluation results, 
and saves the trained model as an artifact.'''

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
mlflow.set_experiment("comparison-complex-experiment")

# Define the model
model = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Define the learning rates to compare
learning_rates = [0.01, 0.001, 0.0001]

# Start a run for each learning rate
for learning_rate in learning_rates:
  with mlflow.start_run(run_name=f"learning_rate={learning_rate}"):
    # Compile the model with the current learning rate

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
    model.fit(X_train, y_train, epochs=5)

    # Evaluate the model on the test set
    eval_results = model.evaluate(X_test, y_test)

    # Log the evaluation results
    for metric_name, metric_value in zip(model.metrics_names, eval_results):
        mlflow.log_metric(metric_name, metric_value)

    # Save the model as an artifact
    mlflow.tensorflow.log_model(model, "model")
