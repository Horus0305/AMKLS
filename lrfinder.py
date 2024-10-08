import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU
from keras import regularizers
from keras.optimizers import Adam

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
X_train = np.random.rand(1000, 32)  # 1000 samples, 32 features
y_train = np.random.rand(1000, 1)    # 1000 target values
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)

# Create your model
model = Sequential([
    Flatten(),
    Dense(1024, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1),  
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-6), loss='mean_squared_error')

# Initialize the Learning Rate Finder
lr_finder = LearningRateFinder(model, train_data, epochs=5, initial_lr=1e-6, final_lr=1)

# Find the optimal learning rate
lr_finder.find_lr()

# Plot the results
lr_finder.plot_lr_vs_loss()
