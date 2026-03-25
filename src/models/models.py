import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.att = nn.Linear(in_dim, 1)

    def forward(self, X):
        # X: [T, N, F]
        scores = self.att(X)                     # [T, N, 1]
        weights = torch.softmax(scores, dim=0)   # over time
        out = (weights * X).sum(dim=0)           # [N, F]
        return out


class TemporalMean(nn.Module):
    def forward(self, X):
        return X.mean(dim=0)


class MLPLinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim * 2, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, 1)

    def forward(self, z, edge_index):
        src, dst = edge_index
        h = torch.cat([z[src], z[dst]], dim=1)
        h = F.relu(self.lin1(h))
        h = torch.sigmoid(self.lin2(h)).view(-1)
        return h


class DotLinkPredictor(nn.Module):
    def forward(self, z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=1)


class TemporalLinkPredictor(nn.Module):
    def __init__(
        self,
        in_dim,
        hid=64,
        predictor="mlp",
        temporal="mean"
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hid = hid

        if temporal == "attention":
            self.temporal = TemporalAttention(in_dim)
        else:
            self.temporal = TemporalMean()

        if predictor == "mlp":
            self.predictor = MLPLinkPredictor(in_dim, hid)
        else:
            self.predictor = DotLinkPredictor()

    def forward(self, X, edge_index):
        """
        X: [T, N, F]
        edge_index: [2, E]
        """

        z = self.temporal(X)            # [N, F]
        pred = self.predictor(z, edge_index)

        return pred, z
