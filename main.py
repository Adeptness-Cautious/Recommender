import torch.nn


def setup():
    # hyperparameters
    batch_size = 128
    learning_rate = 0.0001
    num_epochs = 100
    # How many negative examples per positive example? Paper only says a "relatively small number" to quicken training
    neg_size = 10
    # tradeoff parameter controlling the contributions of the point-wise loss and pair-wise loss
    # based off the paper, 0.68 - 0.71 is a good range
    alpha = 0.695

    # vector user pandas representation
    from dataset import RatingMatrixDataset
    train_set = RatingMatrixDataset(["./ml-100k/u1.base",
                                     "./ml-100k/u2.base",
                                     "./ml-100k/u3.base",
                                     "./ml-100k/u4.base",
                                     "./ml-100k/u5.base"],
                                    neg_size=10)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    # Model instantiation along with optimization constraints
    from model import JNCF
    JNCF = JNCF()
    from torch.optim import Adam
    optimizer = Adam(JNCF.parameters(), lr=learning_rate)

    # Training to now occur
    for epoch in range(num_epochs):

        for i, data in enumerate(train_loader, 0):

            v_u, v_i, item_idx, negative_set, Y_ui = data

            Y_uj = torch.zeros_like(Y_ui)

            for v_j_idx in range(negative_set.shape[1]):
                # Get negative_set
                v_j = negative_set[:, v_j_idx, :]

                # Reset gradients
                optimizer.zero_grad()

                # Get outputs
                y_i, y_j = JNCF(v_u, v_i, v_j)

                # Calculate loss
                # pairwise_loss = torch.sigmoid(y_i - y_j) + torch.sigmoid(torch.pow(y_j, 2))
                # combine = -Y_ui * torch.log(y_i).T
                # loss = (1 - Y_ui) * torch.log(1 - y_i).T
                # loss = alpha * pairwise_loss + (1 - alpha) * (-Y_ui * torch.log(y_i) - (1 - Y_ui) * torch.log(1 - y_i))
                # loss = torch.mean(loss)

                pointwise_i_loss = y_i.T - Y_ui
                pointwise_j_loss = y_j.T - Y_uj

                loss = torch.mean(abs(pointwise_j_loss) + abs(pointwise_i_loss))

                # loss = torch.

                print(loss)

                loss.backward()
                optimizer.step()

    # Testing set
    test_set = RatingMatrixDataset(["./ml-100k/u1.test",
                                    "./ml-100k/u2.test",
                                    "./ml-100k/u3.test",
                                    "./ml-100k/u4.test",
                                    "./ml-100k/u5.test"],
                                   neg_size=10)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False)

    for i, data in enumerate(train_loader, 0):
        v_u, v_i, item_idx, negative_set, Y_ui = data
        Y_uj = torch.zeros_like(Y_ui)
        for v_j_idx in range(negative_set.shape[1]):
            v_j = negative_set[:, v_j_idx, :]
            optimizer.zero_grad()
            y_i, y_j = JNCF(v_u, v_i, v_j)
            pointwise_i_loss = y_i.T - Y_ui
            pointwise_j_loss = y_j.T - Y_uj
            loss = torch.mean(abs(pointwise_j_loss) + abs(pointwise_i_loss))


if __name__ == '__main__':
    setup()
