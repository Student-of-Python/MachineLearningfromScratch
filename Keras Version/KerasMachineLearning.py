"""
Letter Recognition using Convolutional Neural Network (CNN)
-----------------------------------------------------------

This project trains a deep learning model using TensorFlow/Keras to classify
grayscale images of English alphabet characters (A–Z, a–z, and numeric classes if included).
Data is dynamically loaded from a CSV file and structured into training,
validation, and testing pipelines with augmentation applied to improve robustness.

Author: [Your Name]
GitHub: [Your Repository URL]
License: [Choose a license for this project]

Dataset:
--------
Expects a CSV file with:
- 'image' : path to the image file
- 'label' : class label for the image

Model Summary:
--------------
- Convolutional Neural Network with multiple Conv2D + MaxPooling layers
- Dense layers for classification into 62 total classes
- Trained over 20 epochs with real-time augmentation

Outputs:
--------
- Saves prediction results to results.csv
- Displays each test image with a delay for visual inspection
"""

# ==============================
# Imports & System Configuration
# ==============================
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import time

# Enable GPU support if available
# Ensures TensorFlow dynamically manages GPU memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# ==============================
# Load & Prepare Dataset
# ==============================
# Reads labeled dataset metadata from CSV
data = pd.read_csv(r'Data\english.csv')


# ==============================
# Data Generators (Augmentation)
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)


# ==============================
# Train / Validation / Test Split
# ==============================
train, test = train_test_split(
    data,
    test_size=0.2,
    stratify=data['label']
)

train, validation = train_test_split(
    train,
    train_size=7/8,
    stratify=train['label']
)


# ==============================
# Flow Data into Keras Generators
# ==============================
train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    color_mode='grayscale',
    batch_size=16
)

validation_generator = valid_datagen.flow_from_dataframe(
    validation,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    batch_size=16,
    color_mode='grayscale'
)

test_generator = valid_datagen.flow_from_dataframe(
    test,
    x_col='image',
    y_col='label',
    target_size=(100, 100),
    color_mode='grayscale',
    batch_size=1
)


# ==============================
# Model Architecture Definition
# ==============================
"""
Future Improvements:
- Add Dropout layers to reduce overfitting
- Experiment with more efficient architectures (e.g., MobileNet, ResNet)
- Add callbacks for model checkpointing and early stopping
"""
model = Sequential()
model.add(Conv2D(10000, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(5000, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(2500, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1250, activation='relu'))
model.add(Dense(625, activation='relu'))
model.add(Dense(62, activation='softmax'))

model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='categorical_crossentropy'
)


# ==============================
# Model Training
# ==============================
model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator
)


# ==============================
# Generate Predictions
# ==============================
prediction = model.predict(test_generator)
pred_class = np.argmax(prediction, axis=1)

labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
predictions = [labels[k] for k in pred_class]

file_names = test_generator.filenames

results = pd.DataFrame({
    'Filename': file_names,
    'Prediction': predictions
})

results.to_csv('results.csv', index=False)


# ==============================
# Visualize Inference Results
# ==============================
for image in results['Filename']:
    im = Image.open(r'{}'.format(image))
    im.show()
    time.sleep(5)
