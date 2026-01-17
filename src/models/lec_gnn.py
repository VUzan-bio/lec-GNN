from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError:  # pragma: no cover - optional dependency
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]


class LecGNN(nn.Module):
    """GraphSAGE + GAT model with glyco/trafficking readout."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_gat_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.2,
        use_glyco_readout: bool = True,
    ) -> None:
        super().__init__()
        if GATConv is None or SAGEConv is None:
            raise ImportError("torch-geometric is required for LecGNN")

        self.sage = SAGEConv(in_channels, hidden_channels)
        self.gat_layers = nn.ModuleList(
            [
                GATConv(
                    hidden_channels,
                    hidden_channels,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1,
                )
                for _ in range(num_gat_layers)
            ]
        )
        self.gat_out = GATConv(
            hidden_channels,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.use_glyco_readout = use_glyco_readout

        readout_dim = hidden_channels * (2 if use_glyco_readout else 1)
        self.mlp = nn.Sequential(
            nn.Linear(readout_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def _masked_mean_pool(
        x: torch.Tensor, batch: Optional[torch.Tensor], mask: torch.Tensor
    ) -> torch.Tensor:
        if batch is None:
            return x[mask].mean(dim=0, keepdim=True)

        batch_masked = batch[mask]
        x_masked = x[mask]
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros((num_graphs, x.size(1)), device=x.device)
        counts = torch.zeros((num_graphs, 1), device=x.device)
        out.index_add_(0, batch_masked, x_masked)
        counts.index_add_(0, batch_masked, torch.ones((x_masked.size(0), 1), device=x.device))
        counts = counts.clamp(min=1.0)
        return out / counts

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.sage(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        for gat in self.gat_layers:
            if edge_attr is None:
                x = gat(x, edge_index)
            else:
                x = gat(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)

        if edge_attr is None:
            x = self.gat_out(x, edge_index)
        else:
            x = self.gat_out(x, edge_index, edge_attr)
        x = torch.relu(x)

        if node_types is None:
            readout = self._masked_mean_pool(x, batch, torch.ones(x.size(0), dtype=torch.bool, device=x.device))
            return self.mlp(readout).squeeze(-1)

        traffic_mask = node_types == 2
        traffic_pool = self._masked_mean_pool(x, batch, traffic_mask)
        if self.use_glyco_readout:
            glyco_mask = node_types == 1
            glyco_pool = self._masked_mean_pool(x, batch, glyco_mask)
            readout = torch.cat([glyco_pool, traffic_pool], dim=1)
        else:
            readout = traffic_pool
        return self.mlp(readout).squeeze(-1)
