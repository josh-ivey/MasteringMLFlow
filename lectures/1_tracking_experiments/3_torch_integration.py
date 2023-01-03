'''
This file uses MLFlow to track an experiment with PyTorch. 
It first creates an experiment with the name "pytorch-experiment" 
using mlflow.set_experiment("pytorch-experiment"). 
Then, it defines a PyTorch model, loss function, and optimizer.

Next, it loads the MNIST dataset and creates data loaders for the training and test sets.

Then, it starts a run using with mlflow.start_run():. Within this run context, 
it trains the model for 5 epochs, logging the "loss" metric after each epoch.

After training, it evaluates the model on the test set and logs the "test_loss" and 
"test_accuracy" metrics.

Finally, it saves the model as an artifact with the path "model"
using mlflow.pytorch.log_model(model, "model").
'''

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Create an experiment
mlflow.set_experiment("pytorch-experiment")

# Define the model, loss function, and optimizer
model = MLP()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load the MNIST dataset
mnist_train = datasets.MNIST(".", train=True, download=True,
                             transform=transforms.ToTensor())
mnist_test = datasets.MNIST(".", train=False, download=True,
                            transform=transforms.ToTensor())

# Create data loaders for the datasets
train_loader = data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size=64, shuffle=True)

mlflow.set_experiment("pytorch-mnist-experiment")
# Start a run
with mlflow.start_run():
    # Train the model
    for epoch in range(5):
        # Loop over the training data in batches
        for inputs, labels in train_loader:
            # Reshape the input data to be fed into the model
            inputs = inputs.view(-1, 28 * 28)

            # Perform a forward pass through the model
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels)

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update the model weights
            optimizer.step()

            # Log the loss metric
            mlflow.log_metric("loss", loss.item())

    # Evaluate the model on the test set
    with torch.no_grad():
        # Loop over the test data in batches
        for inputs, labels in test_loader:
            # Reshape the input data to be fed into the model
            inputs = inputs.view(-1, 28 * 28)

            # Perform a forward pass through the model
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_fn(outputs, labels)

            # Log the evaluation results
            mlflow.log_metric("test_loss", loss.item())
            mlflow.log_metric("test_accuracy", (outputs.argmax(dim=1) == labels).float().mean())


    # Save the model as an artifact
    mlflow.pytorch.log_model(model, "model")
