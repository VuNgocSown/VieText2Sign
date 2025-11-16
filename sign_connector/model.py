"""Sign Connector model architecture."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class CoordNorm(nn.Module):
    """Normalize 3D coordinates to be centered and scale-invariant."""
    def forward(self, x):
        center = x[:, :, :3].mean(dim=1, keepdim=True)
        x[:, :, :3] = x[:, :, :3] - center
        scale = torch.norm(x[:, :, :3], dim=2, keepdim=True).mean(dim=1, keepdim=True)
        x[:, :, :3] = x[:, :, :3] / (scale + 1e-6)
        return x


class SignConnector(nn.Module):
    """GCN-based model for predicting transition frames between poses."""
    
    def __init__(self, num_joints=46, in_channels=14, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.coord_norm = CoordNorm()
        self.gcn1 = GCNConv(in_channels, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(num_joints * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, edge_index, edge_weight=None):
        B, N, _ = x.shape
        x = self.coord_norm(x)
        outs = []
        for b in range(B):
            h = torch.relu(self.gcn1(x[b], edge_index, edge_weight))
            h = torch.relu(self.gcn2(h, edge_index, edge_weight))
            outs.append(h.flatten())
        h = torch.stack(outs, dim=0)
        return self.fc(h)
