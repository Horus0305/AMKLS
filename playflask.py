from flask import Flask, render_template, request, jsonify
import chess
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Initialize a chess board
board = chess.Board()

# Load the model once to avoid loading it for every move
model = tf.keras.models.load_model('model_512_128_1_other.keras')

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

def play_nn(fen, player='b'):
    board = chess.Board(fen=fen)
    best_move = ''
    worst_move = ''
    minScore = float('inf')
    maxScore = float('-inf')

    for move in board.legal_moves:
        candidate_board = board.copy()
        candidate_board.push(move)
        input_vector = encode_board(str(candidate_board)).astype(np.int32)

        score = model.predict(np.expand_dims(input_vector, axis=0), verbose=0)[0][0]
        if score > maxScore:
            best_move = move
            maxScore = score
        elif score < minScore:
            worst_move = move
            minScore = score

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
