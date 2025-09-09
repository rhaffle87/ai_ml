import model_train as tc
import numpy as np
import pandas as pd
import cuda as cu
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from model_train import nn

#setup device agnostic code
device = "cuda" if tc.cuda.is_available() else "cpu"
print(f"Using {device} device")

# create known parameters
weight = 0.7
bias = 0.3

#create data
start = 0
end = 1
step = 0.02

# generate data
num_samples = 100
X = tc.linspace(start, end, num_samples).unsqueeze(1)   # shape (100, 1)
y = weight * X + bias

# split data (train & test)
X_train, X_test, y_train, y_test = train_test_split(
    X.numpy(), y.numpy(), test_size=0.2, random_state=42
)

# convert to torch tensors
X_train = tc.tensor(X_train, dtype=tc.float32).to(device)
y_train = tc.tensor(y_train, dtype=tc.float32).to(device)
X_test = tc.tensor(X_test, dtype=tc.float32).to(device)
y_test = tc.tensor(y_test, dtype=tc.float32).to(device)

# build linear regression model
model = nn.Linear(in_features=1, out_features=1).to(device)

# define loss and optimizer
loss_fn = nn.MSELoss()
optimizer = tc.optim.SGD(model.parameters(), lr=0.1)

# training loop
epochs = 1000
for epoch in range(epochs):
    # forward pass
    y_pred = model(X_train)

    # compute loss
    loss = loss_fn(y_pred, y_train)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # log every 100 epochs
    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# evaluate on test data
with tc.no_grad():
    y_pred_test = model(X_test)
    test_loss = loss_fn(y_pred_test, y_test)
print(f"Test Loss: {test_loss.item():.6f}")

# visualize results
plt.figure(figsize=(8,5))
plt.scatter(X.numpy(), y.numpy(), label="True data")
plt.scatter(X_test.cpu().numpy(), y_pred_test.cpu().numpy(), label="Predictions", color="red")
plt.legend()
plt.show()

# print learned parameters
print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())
