import pandas as pd
data = pd.read_csv('train.csv')
unique_moves = data['best_move'].unique()
print(len(unique_moves))
