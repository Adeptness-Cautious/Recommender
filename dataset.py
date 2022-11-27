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
            print(data)

        # Normalize rating data
        data['rating'] = (data['rating'] - 1) / 4

        # Create (num of users) 1x(num of movies) vectors to represent users
        import torch
        self.matrix = torch.zeros(943, 1682)
        for index, row in data.iterrows():
            self.matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']
        # for row in self.matrix:
        #     if torch.count_nonzero(row)

    def __getitem__(self, index) -> T_co:
        pass
