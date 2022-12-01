from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
import torch


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
        self.user_matrix = torch.zeros(943, 1682)
        self.item_matrix = torch.zeros(1682, 943)
        for index, row in data.iterrows():
            self.user_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']
            self.item_matrix[int(row['movie_id']) - 1][int(row['user_id'] - 1)] = row['rating']

        # Creating max value of every user rating. Normally 1, but on occasion lower.
        self.user_max = torch.max(self.user_matrix, dim=1)

        # For later
        self.gt = None

    def __getitem__(self, index):
        pass

    # Simple return length
    def __getlen__(self):
        return len(self.user_matrix)

    # Random shuffle also takes the time to prepare the leave-one-out pandas value
    def random_shuffle(self):
        # Shuffle data
        self.user_matrix = self.user_matrix[torch.randperm(self.user_matrix.size()[0])]

        # Doing leave-one-out samples and storing the ground_truth and item_index
        # ground truth to compare answer with, item_index to not create negative of value we're trying to predict
        # gt is rand_gt rank and then index
        torch.set_printoptions(threshold=10000)
        self.gt = torch.zeros(943, 2)
        for index, row in enumerate(self.user_matrix):
            y = torch.where(row != 0)
            z = torch.randperm(y[0].shape[0])
            item_idx = int(y[0][int(z[0])])
            rand_gt = row[item_idx]
            self.gt[index] = torch.tensor([rand_gt, item_idx])
            row[item_idx] = 0

    # For getting a subset of negative examples
    def get_negatives(self):
        pass
