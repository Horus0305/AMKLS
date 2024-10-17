import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# Load your dataset
train_df = pd.read_csv('train.csv', index_col='id')

# Function to encode pieces using material values
def encode_board(board):
    material_dict = {
        'p': -1, 'b': -3.5, 'n': -3, 'r': -5, 'q': -9, 'k': -4,
        'K': 4, '.': 0, 'P': 1, 'B': 3.5, 'N': 3, 'R': 5, 'Q': 9
    }
    board_str = str(board).replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            row_list.append(material_dict.get(piece, 0))  # Use 0 for empty squares
        board_list.append(row_list)
    return np.array(board_list)

# Encode a FEN string into a board representation
def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

# Prepare the training data
X = np.stack(train_df['board'].apply(encode_fen_string)).reshape(-1, 8, 8)  # Shape (num_samples, 8, 8)
y = train_df['black_score'].values

# Normalize y (black_score)
y_mean = y.mean()
y_std = y.std()
y_normalized = (y - y_mean) / y_std

# Normalize X (the board features)
X_mean = X.mean()
X_std = X.std()
X_normalized = (X - X_mean) / X_std

# Build a simple neural network model
model = Sequential([
    Flatten(input_shape=(8, 8)),  # Flatten the 8x8 input to a 1D array
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Output layer for predicting the normalized black score
])

# Compile the model with MSE and MAE as metrics
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error'])

# Train the model
history = model.fit(
    X_normalized, y_normalized,  # Use normalized inputs and target variable
    epochs=100,
    batch_size=10,
    validation_split=0.2  # Use 20% of the data for validation
)

# Save the model
model.save('chess_model.keras')

# Predict on new data (denormalizing predictions)
preds_normalized = model.predict(X_normalized)
preds = preds_normalized * y_std + y_mean  # Denormalize predictions

# Plot training and validation loss
plt.style.use('ggplot')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss During Training')
plt.show()

# Plot Mean Squared Error (MSE)
plt.plot(history.history['mean_squared_error'], label='train MSE')
plt.plot(history.history['val_mean_squared_error'], label='val MSE')
plt.legend()
plt.title('MSE During Training')
plt.show()

# Plot Mean Absolute Error (MAE)
plt.plot(history.history['mean_absolute_error'], label='train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='val MAE')
plt.legend()
plt.title('MAE During Training')
plt.show()
