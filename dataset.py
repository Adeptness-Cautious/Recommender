from torch.utils.data import Dataset


class RatingMatrixDataset(Dataset):
    def __init__(self, fold):
        import pandas as pd
        COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
        train_data = pd.read_csv("./ml-100k/u" + str(fold) + ".base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
        test_data = pd.read_csv("./ml-100k/u" + str(fold) + ".test", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
