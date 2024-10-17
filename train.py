import pandas as pd
import numpy as np
import chess
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout



train_df = pd.read_csv('train.csv', index_col='id')

print(train_df.head())

# Function to split the data into train and validation sets
def split_data(df, val_percentage):
    total_train_samples = len(df)
    val_size = int(total_train_samples * val_percentage)
    
    train_df = df[:-val_size]
    val_df = df[-val_size:]
    
    return train_df, val_df

# Example usage with 10% for validation:
train_df, val_df = split_data(train_df, val_percentage=0.1)

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
            row_list.append(material_dict.get(piece))
        board_list.append(row_list)
    return np.array(board_list)

# Encode a FEN string into a board representation
def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

# Prepare the training and validation data
X_train = np.stack(train_df['board'].apply(encode_fen_string))
y_train = train_df['black_score']

X_val = np.stack(val_df['board'].apply(encode_fen_string))
y_val = val_df['black_score']

# Reshape inputs for CNN (batch_size, height, width, channels)
X_train = X_train.reshape(-1, 8, 8, 1)  # 8x8 board with 1 channel (grayscale)
X_val = X_val.reshape(-1, 8, 8, 1)

# Build CNN model
model = Sequential([
    # First Conv Layer
    Conv2D(32, kernel_size=(3, 3), activation=None, input_shape=(8, 8, 1), kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),  # Keep pooling here
    Dropout(0.3),
    
    # Second Conv Layer (No MaxPooling2D here)
    Conv2D(64, kernel_size=(3, 3), activation=None, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),

    # Flatten the output
    Flatten(),

    # Fully Connected Layers
    Dense(512, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1),  # Output layer
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_val, y_val)
)

# Save the model
model.save('cnn_model_512_128_1_other.keras')

# Plot training and validation loss
plt.style.use('ggplot')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss During Training')
plt.show()
