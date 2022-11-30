from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class RatingMatrixDataset(Dataset):

    # Init functions as a simple standard setup preparation of data by creating the user and item matricies.
    # For implementations that save data, saving the data past this initial init is computationally effective.
    def __init__(self, destinations):
        # Import data
        import pandas as pd
        COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
        data = None
        for destination in destinations:
            if data is None:
                data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
            else:
                new_data = pd.read_csv(destination, sep='\t', names=COLS).astype(int)
                data = pd.concat([data, new_data], axis=0)

        # Normalize rating data
        data['rating'] = data['rating'] / 5
        data = data.reset_index()

        # Create (num of users) 1x(num of movies) vectors to represent users
        import torch
        self.user_matrix = torch.zeros(943, 1682)
        self.item_matrix = torch.zeros(1682, 943)
        for index, row in data.iterrows():
            self.user_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']
            self.item_matrix[int(row['movie_id']) - 1][int(row['user_id'] - 1)] = row['rating']

        # Creating max value of every user rating. Normally 1, but on occasion lower.
        self.user_max = torch.max(self.user_matrix, dim=1)

    def __getitem__(self, index) -> T_co:
        pass

    # Simple return length
    def __getlen__(self):
        return len(self.user_matrix)

    # Random shuffle also takes the time to prepare the leave-one-out pandas value
    def random_shuffle(self):
        # todo: Leave one out, training consists of all ratings except for most recent
        pass

    def get_negatives(self):
        pass
