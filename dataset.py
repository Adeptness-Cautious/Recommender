from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class RatingMatrixDataset(Dataset):
    def __init__(self, destinations):
        # Import data
        import pandas as pd
        COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
        data = None
        for destination in destinations:
            if data is None:
                data = pd.read_csv(destination, sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
            else:
                new_data = pd.read_csv(destination, sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
                data = pd.concat([data, new_data], axis=0)

        # Normalize rating data
        data['rating'] = data['rating'] / 5

        # print(data[data['rating'] == 1].to_string())

        # Create (num of users) 1x(num of movies) vectors to represent users
        import torch
        self.user_matrix = torch.zeros(943, 1682)
        self.item_matrix = torch.zeros(1682, 943)
        for index, row in data.iterrows():
            self.user_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']
            self.item_matrix[int(row['movie_id'] - 1)][int(row['user_id'] - 1)] = row['rating']

        # todo: Leave one out, training consists of all ratings except for most recent

    def __getitem__(self, index) -> T_co:
        pass
