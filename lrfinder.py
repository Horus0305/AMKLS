import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
from keras import regularizers
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder

class LearningRateFinder:
    def __init__(self, model, train_data, epochs=1, initial_lr=1e-6, final_lr=1, beta=0.98):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.beta = beta
        self.lrs = []
        self.losses = []
        self.best_loss = np.inf
        self.avg_loss = 0.0
        self.iteration = 0

    def find_lr(self):
        num = len(self.train_data)
        for epoch in range(self.epochs):
            for x, y in self.train_data:
                self.iteration += 1

                # Compute the learning rate
                lr = self.initial_lr * (self.final_lr / self.initial_lr) ** (self.iteration / (num * self.epochs))
                self.model.optimizer.lr = lr

                # Train the model for one batch
                loss = self.model.train_on_batch(x, y)

                # Ensure loss is a scalar
                loss = loss[0] if isinstance(loss, (list, tuple)) else loss

                # Record the loss
                self.lrs.append(lr)
                self.losses.append(loss)

                # Compute the average loss
                self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
                smooth_loss = self.avg_loss / (1 - self.beta ** self.iteration)

                # Save the best loss
                if smooth_loss < self.best_loss and self.iteration > 1:
                    self.best_loss = smooth_loss
                elif smooth_loss > 4 * self.best_loss:  # Stop if loss diverges
                    return


    def plot_lr_vs_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate (log scale)')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()

# Prepare your dataset (X_train, y_train)
# For demonstration, let's create dummy data
X_train = np.random.rand(1000, 8, 8, 1)  # 1000 samples, 8x8 images with 1 channel
y_train = np.random.randint(0, 3, size=(1000, 1))  # 1000 target classes (0, 1, 2)

# One-hot encode the target values
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train)

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train_encoded)).batch(32)

# Create your model
model = Sequential([
    Conv2D(64, kernel_size=(3, 3), activation=None, input_shape=(8, 8, 1)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # Reduces to 4x4

    Conv2D(128, kernel_size=(2, 2), activation=None),  # Adjust kernel size
    LeakyReLU(alpha=0.1),
    BatchNormalization(),

    Conv2D(256, kernel_size=(2, 2), activation=None),  # Adjust kernel size
    LeakyReLU(alpha=0.1),
    BatchNormalization(),

    GlobalAveragePooling2D(),
    
    Dense(256, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(y_train_encoded.shape[1], activation='softmax')  # Output layer for all unique moves
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize the Learning Rate Finder
lr_finder = LearningRateFinder(model, train_data, epochs=100, initial_lr=1e-6, final_lr=1)

# Find the optimal learning rate
lr_finder.find_lr()

# Plot the results
lr_finder.plot_lr_vs_loss()
