import pandas as pd
import numpy as np
import chess
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout



train_df = pd.read_csv('chess_dataset.csv', index_col='id')

print(train_df.head())

def split_data(df, val_percentage):
    total_train_samples = len(df)
    val_size = int(total_train_samples * val_percentage)
    
    train_df = df[:-val_size]
    val_df = df[-val_size:]
    
    return train_df, val_df

# Example usage with 10% for validation:
train_df, val_df = split_data(train_df, val_percentage=0.1)

def one_hot_encode_peice(piece):
    pieces = list('rnbqkpRNBQKP.')
    arr = np.zeros(len(pieces))
    piece_to_index = {p: i for i, p in enumerate(pieces)}
    index = piece_to_index[piece]
    arr[index] = 1
    return arr

def encode_board(board):
    # first lets turn the board into a string
    board_str = str(board)
    # then lets remove all the spaces
    material_dict = {
        'p': -1,
        'b': -3.5,
        'n': -3,
        'r': -5,
        'q': -9,
        'k': -4,
        'K': 4,
        '.': 0,
        'P': 1,
        'B': 3.5,
        'N': 3,
        'R': 5,
        'Q': 9,
    }
    board_str = board_str.replace(' ', '')
    board_list = []
    for row in board_str.split('\n'):
        row_list = []
        for piece in row:
            # print(piece)
            row_list.append(material_dict.get(piece))
        board_list.append(row_list)
    return np.array(board_list)

encode_board(chess.Board())

def encode_fen_string(fen_str):
    board = chess.Board(fen=fen_str)
    return encode_board(board)

X_train = np.stack(train_df['board'].apply(encode_fen_string))
y_train = train_df['black_score']

X_val = np.stack(val_df['board'].apply(encode_fen_string))
y_val = val_df['black_score']


# With the Keras Sequential model we can stack neural network layers together
model = Sequential([
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1),
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error')

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_val, y_val))

model.save('model_512_128_1_other.keras')



# Lets plot the history of our training session to see how things progressed over time
plt.style.use('ggplot')
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title('Loss During Training')
plt.show()

