"""
MNIST Digit Classification — NumPy Neural Network (from Scratch)
----------------------------------------------------------------

This script implements a 3-layer neural network **without** using deep learning
libraries such as TensorFlow or PyTorch. All operations — including forward and
back-propagation — are done manually using NumPy.

Dataset:
--------
• Uses MNIST handwritten digits dataset from Kaggle format (train.csv)
• Each example is a flattened 28x28 grayscale image → 784 pixels
• 10 output classes (digits 0–9)

Network Design:
---------------
Layer 0  → Input      → 784 nodes
Layer 1  → Hidden     → ReLU activation
Layer 2  → Hidden     → ReLU activation
Layer 3  → Output     → Softmax activation

Training:
---------
• Gradient descent on entire batch (no batching or shuffling per epoch)
• Forward propagation → calculate loss → backward propagation → update weights
• Displays sample prediction every 25 epochs

Limitations:
-----------
• This implementation trains VERY slowly due to full batch + large matrix ops
• No cost function graphing or test evaluation yet
• Improper reuse of training CSV for test set (future fix)

Future Improvements:
--------------------
✔ Mini-batch support
✔ Cross-entropy loss + numeric stability fixes
✔ Weight saving + inference mode
✔ Proper separate test dataset

Author:
-------
[Your Name]
GitHub Repo: [Insert Link Here]
"""

# ==============================
# Imports & Dataset Loading
# ==============================
import numpy as np
import pandas as pd
from PIL import Image

# Load MNIST training CSV file
data_b = pd.read_csv(r'MachineLearningScratch\NumpyMachineLearningDirectory\MNIST Digit Recongizer\train.csv')
test_b = pd.read_csv(r'MachineLearningScratch\NumpyMachineLearningDirectory\MNIST Digit Recongizer\train.csv')

# Duplicate DataFrames to preserve original data
data, test = data_b.copy(), test_b.copy()

# Shuffle dataset to randomize order
data = data.sample(frac=1)

# Separate labels from image data
labels = data.pop('label')

# Convert to NumPy arrays for faster math operations
data, labels = np.array(data), np.array(labels)

# Transpose → shape becomes 784 x 42000 (Pixel x Sample)
data = data.T

image_size, sample_size = data.shape
class_amount = 10  # Digits 0–9 classification


# ==============================
# Parameter Initialization
# ==============================
def define_vars():
    """
    Initialize weight & bias terms for all 3 layers.

    Returns:
        tuple: All weights and constants for each layer
    """
    weight_1 = np.random.rand(class_amount, image_size)
    constant_1 = np.random.rand(class_amount, 1)

    weight_2 = np.random.rand(image_size, class_amount)
    constant_2 = np.random.rand(image_size, 1)

    weight_3 = np.random.rand(class_amount, image_size)
    constant_3 = np.random.rand(class_amount, 1)

    return weight_1, constant_1, weight_2, constant_2, weight_3, constant_3


# ==============================
# Activation Functions
# ==============================
def relu(var):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, var)


def relu_deriv(var):
    """Derivative of ReLU for gradient flow"""
    return var > 0


def softmax(x):
    """Stable softmax for final prediction layer"""
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum().round(3)


# ==============================
# Forward Propagation
# ==============================
def init_layers(layer_0, weight_1, constant_1, weight_2, constant_2, weight_3, constant_3):
    """
    Passes the input through 3 neural network layers.
    Returns linear outputs and activations for each layer.
    """
    linear_1 = np.dot(weight_1, layer_0) + constant_1.T
    layer_1 = relu(linear_1)

    linear_2 = np.dot(weight_2, layer_1) + constant_2
    layer_2 = relu(linear_2)

    linear_3 = np.dot(weight_3, layer_2) + constant_3
    layer_3 = softmax(linear_3)

    return linear_1, layer_1, linear_2, layer_2, linear_3, layer_3


# ==============================
# One-Hot Encoding
# ==============================
def hot_encoding(labels):
    """Convert labels into one-hot encoded matrix form"""
    shape = (labels.size, labels.max() + 1)
    zeros = np.zeros(shape)
    rows = np.arange(labels.size)
    zeros[rows, labels] = 1
    return zeros


# ==============================
# Loss Function + Backpropagation
# ==============================
def loss_func(layer_0, linear_1, layer_1, linear_2, layer_2,
              weight_2, linear_3, layer_3, weight_3, label):
    """
    Custom gradient computation for parameter updates.
    """
    y = hot_encoding(label)

    loss_linear_3 = layer_3 - y.T
    loss_weight_3 = np.dot(loss_linear_3, layer_2.T)
    loss_constant_3 = sum(loss_linear_3)

    loss_linear_2 = np.dot(weight_3.T, linear_3) * relu_deriv(linear_2)
    loss_weight_2 = np.dot(linear_2, layer_1.T)
    loss_constant_2 = sum(loss_linear_2)

    loss_linear_1 = np.dot(weight_2.T, loss_linear_2) * relu_deriv(linear_1)
    loss_weight_1 = np.dot(loss_linear_1, layer_0.T)
    loss_constant_1 = sum(loss_linear_1)

    return loss_weight_3, loss_constant_3, loss_weight_2, loss_constant_2, loss_weight_1, loss_constant_1


# ==============================
# Parameter Update Rule
# ==============================
def redef_vars(loss_weight_3, loss_constant_3, loss_weight_2, loss_constant_2,
               loss_weight_1, loss_constant_1, weight_1, constant_1,
               weight_2, constant_2, weight_3, constant_3, learning_rate):

    weight_1 = weight_1 - learning_rate * loss_weight_1
    constant_1 = constant_1 - learning_rate * loss_constant_1

    weight_2 = weight_2 - learning_rate * loss_weight_2
    constant_2 = constant_2 - learning_rate * loss_constant_2

    weight_3 = weight_3 - learning_rate * loss_weight_3
    constant_3 = constant_3 - learning_rate * loss_constant_3

    # Note: weight_2 transpose here remains consistent with original implementation
    return weight_1, constant_1, weight_2.T, constant_2, weight_3, constant_3


# ==============================
# Prediction + Accuracy
# ==============================
def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(preditions, label):
    print(preditions, label)
    return np.sum(preditions == label) / class_amount


# ==============================
# Training Loop
# ==============================
def machine(image, label, epochs, learning_rate):
    """
    Main training routine.
    Shows sample prediction + image every 25 epochs.
    """
    weight_1, constant_1, weight_2, constant_2, weight_3, constant_3 = define_vars()
    image = image / 255

    for _ in range(epochs):
        linear_1, layer_1, linear_2, layer_2, linear_3, layer_3 = init_layers(
            image, weight_1, constant_1, weight_2, constant_2, weight_3, constant_3
        )

        losses = loss_func(image, linear_1, layer_1, linear_2, layer_2,
                           weight_2, linear_3, layer_3, weight_3, label)
        weight_1, constant_1, weight_2, constant_2, weight_3, constant_3 = redef_vars(
            *losses, weight_1, constant_1, weight_2,
            constant_2, weight_3, constant_3, learning_rate
        )

        if _ % 25 == 0:
            # Display first image to observe evolving predictions
            image = data[:, 0]
            img = np.reshape(image, (28, 28))
            img = Image.fromarray(img.astype('float64'))
            img.show()
            print(f"label: {label[_]}, predicted label: "
                  f"{get_accuracy(preditions=get_predictions(layer_3), label=labels[_])} on iteration {_}")

    return weight_1, constant_1, weight_2, constant_2, weight_3, constant_3


# Start training
weight_1, constant_1, weight_2, constant_2, weight_3, constant_3 = machine(
    data[:], labels, 200, 0.01
)
