from flask import Flask, render_template, request, jsonify
import chess
import numpy as np
import random
import tensorflow as tf

app = Flask(__name__)

# Initialize a chess board
board = chess.Board()

# Load the model once to avoid loading it for every move
model = tf.keras.models.load_model('cnn_model_512_128_1_other.keras')

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

def evaluate_board(board):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    score = 0
    for piece_type in piece_values.keys():
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score

def get_random_move(board):
      # Create a chess board from the FEN string
    legal_moves = list(board.legal_moves)  # Get all legal moves as a list
    
    if legal_moves:  # Check if there are any legal moves
        random_move = random.choice(legal_moves)  # Choose a random legal move
        return str(random_move)  # Return the move as a string
    else:
        return None
    
def play_nn(fen, player='b'):
    board = chess.Board(fen=fen)
    best_move = None
    worst_move = None
    min_score = float('inf')
    max_score = float('-inf')

    for move in board.legal_moves:
        candidate_board = board.copy()
        candidate_board.push(move)
        input_vector = encode_board(candidate_board).astype(np.float32)
        
        # Reshape to match the input shape expected by the model (8x8x1)
        input_vector = input_vector.reshape(1, 8, 8, 1)

        # Predict the board score using the CNN model
        score = model.predict(input_vector, verbose=0)[0][0]
        
        # Update best and worst moves based on the predicted score
        if score > max_score:
            best_move = move
            max_score = score
        if score < min_score:
            worst_move = move
            min_score = score

    if not best_move:
        best_move = get_random_move(board)
    if not worst_move:
        worst_move = get_random_move(board)

    # Return the worst move for 'b' (AI playing as black), or best move otherwise
    return str(worst_move) if player == 'b' else str(best_move)

@app.route('/')
def index():
    return render_template('index.html', board_svg=board._repr_svg_())

@app.route('/make_move', methods=['POST'])
def make_move():
    global board
    user_move = request.json.get('move')
    
    if user_move and user_move in [str(move) for move in board.legal_moves]:
        board.push_san(user_move)
        
        if board.outcome() is not None:
            return jsonify({'board_svg': board._repr_svg_(), 'game_over': True})

        ai_move = play_nn(board.fen(), player='b')
        board.push_san(ai_move)

        return jsonify({
            'board_svg': board._repr_svg_(),
            'ai_move': ai_move,
            'game_over': board.outcome() is not None
        })
    
    return jsonify({'error': 'Invalid move'})

if __name__ == '__main__':
    app.run(debug=True)
