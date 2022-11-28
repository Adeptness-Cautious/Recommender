import torch.nn as nn


class JNCF(nn.Module):

    def __init__(self):
        super(JNCF, self).__init__()
        self.df_user_1 = nn.Linear(943, 256)
        self.df_user_2 = nn.Linear(256, 128)
        self.df_user_3 = nn.Linear(128, 64)

        self.df_item_1 = nn.Linear(1682, 256)
        self.df_item_2 = nn.Linear(256, 128)
        self.df_item_3 = nn.Linear(128, 64)

        self.dl_1 = nn.Linear(128, 8)

    def forward(self, v_u, v_i, v_j):
        import torch.nn.functional as F
        # User feature learning
        v_u = F.relu(self.df_user_1(v_u))
        v_u = F.relu(self.df_user_2(v_u))
        z_u = F.relu(self.df_user_3(v_u))

        # Item feature learning
        v_i = F.relu(self.df_item_1(v_i))
        v_i = F.relu(self.df_item_2(v_i))
        z_i = F.relu(self.df_item_3(v_i))

        v_j = F.relu(self.df_item_1(v_j))
        v_j = F.relu(self.df_item_2(v_j))
        z_j = F.relu(self.df_item_3(v_j))

        # Concatenate both feature learning matrices
        import torch
        a_ui = torch.cat((z_u, z_i), 1)
        a_uj = torch.cat((z_u, z_j), 1)

        # Combined matrices
        y_ui = F.relu(self.dl_1(a_ui))
        y_uj = F.relu(self.dl_1(a_uj))

        return y_ui, y_uj