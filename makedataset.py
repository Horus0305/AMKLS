import pandas as pd
import chess.pgn
import chess.engine

# Initialize chess engine
engine = chess.engine.SimpleEngine.popen_uci(r"D:\AMKLS\stockfish\stockfish-windows-x86-64-avx2.exe")

data = []
entry_count = 0  # Initialize entry count

# Load PGN file
def filter_games(file_path, min_rating=2200):
    with open(file_path) as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            # Extract player ratings
            white_rating = int(game.headers.get('WhiteElo', 0))
            black_rating = int(game.headers.get('BlackElo', 0))

            # Check if either player's rating is above the threshold
            if white_rating > min_rating or black_rating > min_rating:
                yield game

# Process the filtered games
for game in filter_games('selectedgames.pgn'):
    board = chess.Board()
    for move in game.mainline_moves():
        board.push(move)
        board_fen = board.fen()  # Get FEN string

        try:
            result = engine.analyse(board, chess.engine.Limit(time=1))  # Analyze position
            
            # Check if 'score' and 'pv' are in the result
            black_score = result['score'].relative.score() if 'score' in result else None
            
            if 'pv' in result and result['pv']:
                black_score = result['score'].relative.score() if 'score' in result else None
                best_move = result['pv'][0]  # Get the best move

                # Append the data only if 'pv' is present and not empty
                data.append({
                    'id': len(data),  # Use current length as ID
                    'board': board_fen,
                    'black_score': black_score,
                    'best_move': best_move.uci()
                })
                entry_count += 1
                print(entry_count)

        except Exception as e:
            print(f"An error occurred while analyzing the position: {e}")

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('chess_dataset.csv', index=False)
engine.quit()