"""
Handwritten English Character Classifier — Custom Neural Network (NumPy Only)
-----------------------------------------------------------------------------

This script demonstrates a fully manual neural network implementation
(without using any deep learning frameworks) to classify grayscale English
alphabet characters. It includes:

- Custom forward and backpropagation operations
- ReLU and Softmax activation functions
- Manual parameter updates using gradient descent
- On-the-fly single-image data processing to reduce memory load

Dataset Requirement:
-------------------
CSV file must include:
- 'image' : path to image file
- 'label' : class label (A–Z, a–z, digits, etc.)

Notes:
------
• Each image is converted to grayscale and reshaped into a flattened vector.  
• Entire forward + backward pass is done on a single image due to memory constraints.  
• Model parameters are randomly initialized and updated over multiple iterations.  
• This implementation is intended for learning and educational exploration.

Future Enhancements:
--------------------
• Batch processing for improved training accuracy  
• Add evaluation metrics over multiple samples  
• Introduce weight regularization & learning rate scheduling  
• Save trained weights for inference later  
"""

# ==============================
# Imports & Dataset Loading
# ==============================
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import time
from sklearn.utils import shuffle

s = time.time()

data = pd.read_csv(r'Downloads\english.csv')

# Create one-hot encoded label table for classification
labels = pd.get_dummies(data['label'])

# Reorder columns, then shuffle dataset for randomness
columns = list(data.columns.values)
data = data[columns[::-1]]
data = shuffle(data)


# ==============================
# Image Processing Functions
# ==============================
def image_to_pixel(image_loc):
    """
    Load an image file, convert to grayscale, and return as a NumPy pixel array
    along with its label.

    Parameters:
        image_loc (str): Path to image file

    Returns:
        np.ndarray: Flattened grayscale pixel array
        label (str): Character label for the image
    """
    try:
        n_image = Image.open('{}'.format(image_loc))
        g_image = ImageOps.grayscale(n_image)
        label = data.loc[data['image'] == image_loc, 'label'].item()
        return np.array(g_image), label
    except FileNotFoundError:
        return exit(f'File Location \n({image_loc})\n Invalid')


def processing_next_image():
    """
    Generator used to load images one-by-one during training to minimize
    memory consumption.

    Yields:
        tuple: (image_pixel_array, label)
    """
    for images in data['image']:
        yield image_to_pixel(images)


# Load only first image for initial model setup
image, label = next(processing_next_image())
m, n = np.shape(image)
image = np.reshape(image, m*n)  # Flatten pixel matrix

# Number of classification categories (62: A-Z, a-z, digits)
c_num = 62


# ==============================
# Activation Functions
# ==============================
def ReLu(z):
    """Rectified Linear Unit activation"""
    return np.maximum(z, 0)


def softmax(x):
    """Softmax activation for multi-class probability distribution"""
    return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum().round(2)


# ==============================
# Weight Initialization
# ==============================
def initial_params():
    """
    Initialize network parameters:
    Layer 1: Fully connected
    Layer 2: Output layer
    """
    w1 = np.random.rand(c_num, m*n) - 0.5
    b1 = np.random.rand(c_num, 1) - 0.5
    w2 = np.random.rand(c_num, c_num) - 0.5
    b2 = np.random.rand(c_num, 1) - 0.5
    return w1, b1, w2, b2


w1, b1, w2, b2 = initial_params()


# ==============================
# Forward Propagation
# ==============================
def forward_prop(w1, b1, w2, b2, image):
    z1 = w1.dot(image) + b1.T
    z1 = z1.T
    a1 = ReLu(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, z2, a1, a2


# ==============================
# Derivative of ReLU
# ==============================
def D_ReLu(z):
    return z > 0


# ==============================
# One-Hot Encoding
# ==============================
def one_hot(label_char):
    """
    Convert label into one-hot vector form.
    Extracts one sample per class (assuming 55 images per class).
    """
    label_ar = labels[label_char]
    label_ar = label_ar[::55]
    return np.array(label_ar).T


# ==============================
# Backward Propagation
# ==============================
def back_prop(z1, a1, z2, a2, w1, w2, image, y=label):
    y = one_hot(y)
    y = y.reshape((62, 1))
    image = image.reshape((1, 1080000))

    dz2 = a2 - y
    dw2 = dz2.dot(a1.T)
    db2 = np.sum(dz2)
    dz1 = w2.T.dot(dz2) * D_ReLu(z1)
    dw1 = dz1.dot(image)
    db1 = np.sum(dz1)
    return dw1, db1, dw2, db2


# ==============================
# Gradient Descent Parameter Update
# ==============================
def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2
    return w1, b1, w2, b2


# ==============================
# Prediction & Accuracy Checking
# ==============================
def get_predictions(a2):
    return np.argmax(a2, 0)


def get_accuracy(preditions):
    print(preditions, label)
    return np.sum(preditions == label) / 62


# ==============================
# Training Loop — Single Image Test
# ==============================
def gradient_descent(image, label, alpha, iters):
    """
    Perform gradient descent on a single training example.

    Parameters:
        alpha (float): learning rate
        iters (int): number of iterations

    Returns:
        Updated weight and bias matrices
    """
    w1, b1, w2, b2 = initial_params()
    for i in range(iters):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, image)
        dw1, db1, dw2, db2 = back_prop(z1, a1, z2, a2, w1, w2, image, label)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print('Iteration: ', i)
            prediction = get_predictions(a2)
            print(get_accuracy(prediction))
    return w1, b1, w2, b2


# Execute Training
w1, b1, w2, b2 = gradient_descent(image, label, 0.3, 200)
