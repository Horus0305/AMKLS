import numpy as np
import chess
import tensorflow as tf
from IPython.display import SVG, display

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
    # Define piece values
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0  # King's value is generally not considered in material evaluation
    }

    score = 0
    for piece_type in piece_values.keys():
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score

def play_nn(fen, player='b'):
    # We can create a python-chess board instance from the FEN string like this:
    board = chess.Board(fen=fen)

    # And then evaluate all legal moves
    best_move = ''
    worst_move = ''
    minScore = float('inf')
    maxScore = float('-inf')

    moves = []
    for move in board.legal_moves:
        # For each move, we'll make a copy of the board and try that move out
        candidate_board = board.copy()
        candidate_board.push(move)
        input_vector = encode_board(str(candidate_board)).astype(np.int32)

        # This is where our model gets to shine! It tells us how good the resultant score board is for black:
        model = tf.keras.models.load_model('model_512_128_1_other.keras')
        score = model.predict(np.expand_dims(input_vector, axis=0), verbose=0)[0][0]
        if score > maxScore:
            best_move = move
            maxScore = score
        elif score < minScore:
            worst_move = move
            minScore = score
        # moves.append((score, move))
        # if show_move_evaluations:
        #     print(f'{move}: {score}')

    # By default sorting our moves will put the lowest scores at the top.
    # This would give us the right answer if we were playing as white,
    # but if we're playing as black we want to reverse things (then grab the first move):

    if(player=='b'):
        return str(worst_move)
    # Now we turn our move into a string, return it and call it a day!
    return str(best_move)



# Our play function accepts whatever strategy our AI is using, like play_random from above
def play_game(ai_function):
    board = chess.Board()

    while board.outcome() is None:
        display(SVG(board._repr_svg_()))

        score = evaluate_board(board)
        print(f'Current Score - White: {score} | Black: {-score}')

        if board.turn == chess.WHITE:
            user_move = input('Your move: ')
            if user_move == 'quit':
                break
            while user_move not in [str(move) for move in board.legal_moves]:
                print('That wasn\'t a valid move. Please enter a move in Standard Algebraic Notation')
                user_move = input('Your move: ')
            board.push_san(user_move)

        elif board.turn == chess.BLACK:
            ai_move = ai_function(board.fen())
            if ai_move is None:
                print("AI has no valid moves.")
                break
            print(f'AI move: {ai_move}')
            board.push_san(ai_move)

    print(board.outcome())

play_game(play_nn)