def setup():
    # hyperparameters
    batch_size = 256
    learning_rate = 0.0001

    # vector user pandas representation
    from dataset import RatingMatrixDataset
    user_set_train, item_set_train = RatingMatrixDataset(["./ml-100k/u1.base",
                                                          "./ml-100k/u2.base",
                                                          "./ml-100k/u3.base",
                                                          "./ml-100k/u4.base",
                                                          "./ml-100k/u5.base"])
    user_set_test, item_set_test = RatingMatrixDataset(["./ml-100k/u1.test",
                                                        "./ml-100k/u2.test",
                                                        "./ml-100k/u3.test",
                                                        "./ml-100k/u4.test",
                                                        "./ml-100k/u5.test"])

    # todo: Metrics are hit ratio (is in top n?)
    #  and normalized discount cumulative gain (is ranked at top?)



if __name__ == '__main__':
    setup()
