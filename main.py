import numpy as np
import torch.nn


def JNCF():
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
    print("Creating training dataset.")
    from dataset import RatingMatrixDataset
    train_set = RatingMatrixDataset(["./ml-100k/u1.base",
                                     "./ml-100k/u2.base",
                                     "./ml-100k/u3.base",
                                     "./ml-100k/u4.base",
                                     "./ml-100k/u5.base"],
                                    neg_size=neg_size)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True)

    # Model instantiation along with optimization constraints
    from model import JNCF
    JNCF = JNCF()
    from torch.optim import Adam
    optimizer = Adam(JNCF.parameters(), lr=learning_rate)

    point_loss_function = torch.nn.BCEWithLogitsLoss()

    # Training to now occur
    print("Training")
    total_loss = [0] * num_epochs
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

                # HR@10:21%
                point_loss = abs(point_loss_function(y_i.reshape(y_i.shape[0]), Y_ui.reshape(Y_ui.shape[0])))
                pair_loss = torch.mean((torch.sigmoid(y_j - y_i) + torch.sigmoid(torch.pow(y_j, 2))))
                loss = alpha * pair_loss + (1 - alpha) * point_loss

                # Record loss
                total_loss[epoch] += loss

                # Backpropagate
                loss.backward()
                optimizer.step()

    # Code to show the training
    print(total_loss)
    plot_loss = list()
    for tensor in total_loss:
        plot_loss.append(tensor.item())
    import matplotlib.pyplot as plt
    plt.plot(plot_loss)
    plt.show()

    # Evaluating time
    print("Creating evaluation dataset.")
    test_set = RatingMatrixDataset(["./ml-100k/u1.test",
                                    "./ml-100k/u2.test",
                                    "./ml-100k/u3.test",
                                    "./ml-100k/u4.test",
                                    "./ml-100k/u5.test"],
                                   neg_size=10)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=True)

    # Calculating hit rate: Leave one out, feed in all items, then see if that one out is something
    # we recommended in the top 10 it's a hit.
    hit = 0
    miss = 0
    print("Evaluating")
    user_matrix, item_matrix = test_set.get_matricies()
    for i, data in enumerate(test_loader, 0):

        v_u, v_i, item_idx, negative_set, Y_ui = data

        item_ratings = np.zeros([item_matrix.shape[0]])

        # The paper only rates 100 items per user
        items_rated = 0

        for i_idx, item in enumerate(item_matrix):

            # Feed in all 0s and the rated item
            if user_matrix[i][i_idx] == 0 or i_idx == item_idx:

                item = item.reshape([1, 943])

                y_i, _ = JNCF(v_u, item, item)

                if y_i == torch.tensor(0):
                    y_i = torch.tensor(-1)

                item_ratings[i_idx] = y_i

            # If the 100th
            if items_rated == 99:
                item = item_matrix[item_idx]
                item = item.reshape([1, 943])
                y_i, _ = JNCF(v_u, item, item)
                if y_i == torch.tensor(0):
                    y_i = torch.tensor(-1)
                item_ratings[item_idx] = y_i
                break

            items_rated += 1

        sort_index = np.argpartition(item_ratings, -10)[-10:]
        top_sorted = sort_index[np.argsort(item_ratings[sort_index])]

        if int(item_idx.item()) in top_sorted:
            hit += 1
        else:
            miss += 1

        if i == 100:
            break

    print("hits: " + str(hit))
    print("miss: " + str(miss))


if __name__ == '__main__':
    JNCF()
