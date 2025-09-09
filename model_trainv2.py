import torch as tc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import nn

# from torch.utils.data import DataLoader, TensorDataset
tc.manual_seed(42)

# setup device agnostic code
device = "cuda" if tc.cuda.is_available() else "cpu"
print(f"Using {device} device")

# create known parameters
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
x = tc.arange(start, end, step).unsqueeze(dim=1)  # shape (50, 1)
Y = weight * x + bias + tc.sqrt(tc.tensor(0.01)) * tc.randn_like(x)

print(f'Y dimension: {Y.shape}, Y: {Y[:5]}')
print(f'x dimension: {x.shape}, x: {x[:5]}')

train_ratio = 0.8
test_ratio = 1 - train_ratio

#split data into train and test sets
x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=test_ratio, random_state=42)

#function to plot predictions
def plot_predictions(train_data=x_train,
                     train_labels=Y_train,
                     test_data=x_test,
                     test_labels=Y_test,
                     predictions=None):
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
plot_predictions()