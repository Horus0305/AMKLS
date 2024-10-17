import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chess
from keras.models import load_model

# Load the trained model
model = load_model('chess_model.keras')

# Load the training dataset
train_df = pd.read_csv('train.csv', index_col='id')

# Function to encode pieces using material values (same as used for training)
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

def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

# Encode FEN strings in the training set into board representations
X_train = np.stack(train_df['board'].apply(encode_fen_string)).reshape(-1, 8, 8)
y_train = train_df['black_score'].values

# Normalize y_train (black_score) using the same mean and std
y_mean = np.mean(y_train)
y_std = np.std(y_train)
y_train_normalized = (y_train - y_mean) / y_std

# Evaluate the model on the training set
train_loss, train_mse, train_mae = model.evaluate(X_train, y_train_normalized)

# Print training metrics
print(f"Training Loss: {train_loss:.4f}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Training MAE: {train_mae:.4f}")

# Make predictions on the training set
# Make predictions on the training set
preds_normalized = model.predict(X_train)
preds = preds_normalized * y_std + y_mean  # Denormalize predictions

# Flatten predictions to match the shape of y_train
preds_flattened = preds.flatten()

# Compare predictions to actual values
plt.figure(figsize=(12, 6))
plt.scatter(y_train, preds_flattened, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # Line of equality
plt.title('Model Predictions vs Actual Black Score on Training Set')
plt.xlabel('Actual Black Score')
plt.ylabel('Predicted Black Score')
plt.xlim([y_train.min(), y_train.max()])
plt.ylim([y_train.min(), y_train.max()])
plt.grid()
plt.show()

# Calculate and print correlation
correlation = np.corrcoef(y_train, preds_flattened)[0, 1]
print(f"Correlation between actual and predicted values on training set: {correlation:.4f}")

