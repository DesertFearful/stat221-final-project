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
    def __init__(self, data_dim, hidden_dim=64, hidden_dims=None, feature_map="raw"):
        super().__init__()
        self.data_dim = data_dim
        self.feature_map = feature_map

        if feature_map not in {"raw", "quadratic"}:
            raise ValueError(f"Expected feature_map to be 'raw' or 'quadratic', got {feature_map}")

        if hidden_dims is None:
            hidden_dims = (hidden_dim,)

        layers = []
        in_dim = self.feature_dim()

        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def feature_dim(self):
        if self.feature_map == "raw":
            return self.data_dim

        return 2 * self.data_dim + self.data_dim * (self.data_dim - 1) // 2

    def make_features(self, x):
        if self.feature_map == "raw":
            return x

        squared = x * x - 1.0
        interactions = []

        for i in range(self.data_dim):
            for j in range(i + 1, self.data_dim):
                interactions.append((x[:, i] * x[:, j]).unsqueeze(1))

        if interactions:
            interaction_features = torch.cat(interactions, dim=1)
            return torch.cat([x, squared, interaction_features], dim=1)

        return torch.cat([x, squared], dim=1)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.data_dim:
            raise ValueError(f"Expected x to have shape (batch, {self.data_dim}), got {tuple(x.shape)}")

        features = self.make_features(x)
        return self.net(features).squeeze(-1)
