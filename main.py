import torch.nn


def setup():
    # hyperparameters
    batch_size = 256
    learning_rate = 0.0001
    num_epochs = 100
    # tradeoff parameter controlling the contributions of the point-wise loss and pair-wise loss
    # based off the paper, 0.68 - 0.71 is a good range
    alpha = 0.695

    # vector user pandas representation
    from dataset import RatingMatrixDataset
    train_set = RatingMatrixDataset(["./ml-100k/u1.base",
                                                          "./ml-100k/u2.base",
                                                          "./ml-100k/u3.base",
                                                          "./ml-100k/u4.base",
                                                          "./ml-100k/u5.base"])
    test_set = RatingMatrixDataset(["./ml-100k/u1.test",
                                                        "./ml-100k/u2.test",
                                                        "./ml-100k/u3.test",
                                                        "./ml-100k/u4.test",
                                                        "./ml-100k/u5.test"])

    # Model instantiation
    from model import JNCF
    JNCF = JNCF()
    from torch.optim import Adam
    optimizer = Adam(JNCF.parameters(), lr=learning_rate)

    # Training to now occur
    for epoch in range(num_epochs):

        # todo: Random shuffle of set that user rates
        user_set, item_set = train_set.random_shuffle()

        for user in user_set:

            # todo: Sample negative
            negative_set = train_set.get_negatives(user)

            for negative in negative_set:

                # Reset gradients
                optimizer.zero_grad()

                # Get outputs
                y_i, y_j = JNCF(v_u, v_i, v_j)

                # Calculate loss
                pairwise_loss = torch.sigmoid(y_i - y_j) + torch.sigmoid(torch.pow(y_j, 2))
                Y_ui = ground_truth / max_rating
                L = alpha * pairwise_loss + (1 - alpha) * (-Y_ui * torch.log(y_i) - (1 - Y_ui) * torch.log(1 - y_i))




    # todo: Metrics are hit ratio (is in top n?)
    #  and normalized discount cumulative gain (is ranked at top?)


if __name__ == '__main__':
    setup()
