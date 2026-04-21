import torch
from torch import nn


def _activation_module(name):
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2)

    raise ValueError(f"Unsupported activation '{name}'")


def _resolve_hidden_dims(hidden_dim, hidden_dims, depth, default_depth):
    if hidden_dims is not None:
        resolved = tuple(int(value) for value in hidden_dims)
    else:
        if depth is None:
            depth = default_depth
        resolved = tuple(int(hidden_dim) for _ in range(depth))

    for value in resolved:
        if value < 1:
            raise ValueError(f"Hidden dimensions must be positive, got {resolved}")
    return resolved


def _make_mlp(input_dim, hidden_dims, output_dim, activation):
    layers = []
    in_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(_activation_module(activation))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class FactorizedGenerator(nn.Module):
    def __init__(
        self,
        data_dim,
        latent_dim=1,
        hidden_dim=64,
        hidden_dims=None,
        depth=None,
        activation="silu",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims, depth, default_depth=3)
        self.coord_nets = nn.ModuleList(
            [_make_mlp(latent_dim, self.hidden_dims, 1, activation=activation) for _ in range(data_dim)]
        )

    def forward(self, z):
        if z.ndim != 3:
            raise ValueError(f"Expected z to have shape (batch, data_dim, latent_dim), got {tuple(z.shape)}")
        if z.shape[1] != self.data_dim or z.shape[2] != self.latent_dim:
            raise ValueError(f"Expected z.shape[1:] == ({self.data_dim}, {self.latent_dim}), got {tuple(z.shape[1:])}")

        coords = []
        for j, net in enumerate(self.coord_nets):
            coords.append(net(z[:, j, :]))
        return torch.cat(coords, dim=1)


class Critic(nn.Module):
    def __init__(
        self,
        data_dim,
        hidden_dim=64,
        hidden_dims=None,
        depth=None,
        feature_map="raw",
        activation="leaky_relu",
    ):
        super().__init__()
        self.data_dim = data_dim
        self.feature_map = feature_map

        if feature_map not in {"raw", "quadratic"}:
            raise ValueError(f"Expected feature_map to be 'raw' or 'quadratic', got {feature_map}")

        resolved_hidden_dims = _resolve_hidden_dims(hidden_dim, hidden_dims, depth, default_depth=2)
        self.net = _make_mlp(self.feature_dim(), resolved_hidden_dims, 1, activation=activation)

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
