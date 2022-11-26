from torch.utils.data import Dataset

class UserRatingDataset(Dataset):


def setup():
    import pandas as pd
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    train_data = pd.read_csv("./ml-100k/u1.base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
    test_data = pd.read_csv("./ml-100k/u1.test", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
    print(train_data)

    # vector user pandas representation
    v_u_pd =


if __name__ == '__main__':
    setup()


