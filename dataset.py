from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class RatingMatrixDataset(Dataset):
    def __init__(self, fold):
        # Import data
        import pandas as pd
        COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
        train_data = pd.read_csv("./ml-100k/u" + str(fold) + ".base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
        test_data = pd.read_csv("./ml-100k/u" + str(fold) + ".test", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
        # Normalize rating data
        train_data['rating'] = (train_data['rating'] - 1) / 4
        test_data['rating'] = (test_data['rating'] - 1) / 4

        # Create (num of users) 1x(num of movies) vectors to represent users
        import numpy as np
        train_matrix = np.empty([943, 1682])
        test_matrix = np.empty([462, 1682])
        for index, row in train_data.iterrows():
            train_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']
        for index, row in test_data.iterrows():
            test_matrix[int(row['user_id']) - 1][int(row['movie_id']) - 1] = row['rating']

    def __getitem__(self, index) -> T_co:
        pass