import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SnapshotTemporalGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gnn_hidden_dim: int = 128,
        rnn_hidden_dim: int = 128,
        gnn_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        predictor: str = "mlp",
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.predictor = predictor
        self.dropout = dropout

        self.input_proj = nn.Linear(in_dim, gnn_hidden_dim)
        self.gnn_convs = nn.ModuleList([GCNConv(gnn_hidden_dim, gnn_hidden_dim)])
        for _ in range(gnn_layers - 1):
            self.gnn_convs.append(GCNConv(gnn_hidden_dim, gnn_hidden_dim))

        self.gru = nn.GRU(
            input_size=gnn_hidden_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=1,
        )
        self.att_proj = nn.Linear(rnn_hidden_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 4, rnn_hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(dropout),
            nn.Linear(rnn_hidden_dim, 1),
        )

    def encode_snapshot(self, x_t: torch.Tensor, edge_index_t: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x_t)
        for i, conv in enumerate(self.gnn_convs):
            h = conv(h, edge_index_t)
            if i < len(self.gnn_convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return torch.nan_to_num(h, nan=0.0, posinf=5.0, neginf=-5.0)

    def encode(self, X_seq: torch.Tensor, edge_index_seq) -> tuple[torch.Tensor, torch.Tensor]:
        snapshot_embs = [self.encode_snapshot(X_seq[t], edge_index_seq[t]) for t in range(X_seq.size(0))]
        H = torch.stack(snapshot_embs, dim=0)
        H_temporal, _ = self.gru(H)

        if self.use_attention:
            att = torch.clamp(self.att_proj(H_temporal), -10, 10)
            scores = torch.softmax(att, dim=0)
            z = (H_temporal * scores).sum(dim=0)
        else:
            z = H_temporal.mean(dim=0)

        z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
        return H_temporal, z

    def decode_logits(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.empty(0, device=z.device)

        src, dst = edge_index
        s = z[src]
        d = z[dst]

        if self.predictor == "dot":
            return (s * d).sum(dim=1)

        feat = torch.cat([s, d, torch.abs(s - d), s * d], dim=1)
        return self.mlp(feat).view(-1)

    def decode_proba(self, z: torch.Tensor, edge_index: torch.Tensor, clip: float = 8.0) -> torch.Tensor:
        logits = self.decode_logits(z, edge_index)
        logits = torch.clamp(logits, -clip, clip)
        return torch.sigmoid(logits)
