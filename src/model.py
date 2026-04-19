import torch
from torch import nn


class FactorizedGenerator(nn.Module):
    def __init__(self, data_dim, latent_dim=1, hidden_dim=64):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.coord_nets = nn.ModuleList()

        for _ in range(data_dim):
            net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), 
                                nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, 1))
            self.coord_nets.append(net)

    def forward(self, z):
        if z.ndim != 3:
            raise ValueError(f"Expected z to have shape (batch, data_dim, latent_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.data_dim or z.shape[2] != self.latent_dim:
            raise ValueError(f"Expected z.shape[1:] == ({self.data_dim}, {self.latent_dim}), got {tuple(z.shape[1:])}")
        
        coords = []
        for j, net in enumerate(self.coord_nets):
            x_j = net(z[:, j, :])
            coords.append(x_j)
        return torch.cat(coords, dim=1)
    

class Critic(nn.Module):
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.data_dim = data_dim
        self.net = nn.Sequential(nn.Linear(data_dim, hidden_dim),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(hidden_dim, 1))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"Expected x to have shape (batch, {self.data_dim}), got {tuple(x.shape)}")

        return self.net(x).squeeze(-1)