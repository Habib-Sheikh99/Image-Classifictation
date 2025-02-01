import pandas as pd
import numpy as np

from tensorflow import keras as krs
from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Flatten, Dense

from sklearn.model_selection import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset///
mnist = krs.datasets.mnist 

# Split into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Preprocess the data
# Normalizing...
x_train, x_test = x_train / 255.0, x_test / 255.0  # This will normalize the Training and Testing Data
                                                   # It will convert the data into the range of 0 - 1 
# Step 3: Build the model
from tensorflow.keras.layers import Flatten, Dense
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into a 1D array
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax')  # Output layer with 10 neurons (one for each class)
])
# Step 4: Compile the model
from tensorflow.keras.losses import SparseCategoricalCrossentropy as SpCtgCrossEntropy
from tensorflow.keras.optimizers import Adam as adam
model.compile(
  optimizer=adam(learning_rate=0.004), 
  loss=SpCtgCrossEntropy(), 
  metrics=['accuracy'],
)
# Step 5: Train the model
model.fit(x_train, y_train, epochs=5)

# Step 6: Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_accuracy:.2f}\n\n")

# Step 7: Visualize predictions
predictions = model.predict(x_test)

# Display a few test images with their predicted and true labels
plt.figure(figsize=(10, 10), facecolor='gray') # Specify the Size and Foreground Color according to preference or requirements.
for i in range(9):
    plt.subplot(3, 3, i + 1)  # Organizing..
    plt.imshow(x_test[i], cmap='gray') # This is the Image in the Visulaiztion Window.
    plt.title(f"Pred: {predictions[i].argmax()}, True: {y_test[i]}") # And These are the labels on those images.
    plt.axis('off') # If you need explanation on this, Go to a Doctor! 
plt.show()

# Analyze  the Model to Make Changes, *(Ofcourse!!!!!!!!)

