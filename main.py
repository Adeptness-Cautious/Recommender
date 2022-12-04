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

    # Valid set
    valid_set = RatingMatrixDataset(["./ml-100k/u1.test",
                                     "./ml-100k/u2.test",
                                     "./ml-100k/u3.test",
                                     "./ml-100k/u4.test",
                                     "./ml-100k/u5.test"],
                                    neg_size=10)

    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size,
                                               shuffle=False)

    # Model instantiation along with optimization constraints
    from model import JNCF
    JNCF = JNCF()
    from torch.optim import Adam
    optimizer = Adam(JNCF.parameters(), lr=learning_rate)

    point_loss_function = torch.nn.BCEWithLogitsLoss()

    total_loss = [0] * num_epochs
    # Training to now occur
    for epoch in range(num_epochs):

        for i, data in enumerate(train_loader, 0):

            v_u, v_i, item_idx, negative_set, Y_ui = data

            for v_j_idx in range(negative_set.shape[1]):
                # Get negative_set
                v_j = negative_set[:, v_j_idx, :]

                # Reset gradients
                optimizer.zero_grad()

                # Get outputs
                y_i, y_j = JNCF(v_u, v_i, v_j)

                # Calculate loss
                point_loss = abs(point_loss_function(y_i.reshape(y_i.shape[0]), Y_ui.reshape(Y_ui.shape[0])))
                pair_loss = torch.mean((torch.sigmoid(y_j - y_i) + torch.sigmoid(torch.pow(y_j, 2))))
                loss = alpha * pair_loss + (1 - alpha) * point_loss

                # Record loss
                total_loss[epoch] += loss

                # Backpropagate
                loss.backward()
                optimizer.step()
    print(total_loss)

    plot_loss = list()
    for tensor in total_loss:
        plot_loss.append(tensor.item())

    import matplotlib.pyplot as plt
    plt.plot(plot_loss)
    plt.show()


if __name__ == '__main__':
    setup()
