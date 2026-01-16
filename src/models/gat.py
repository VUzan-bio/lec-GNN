from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError:  # pragma: no cover - optional at runtime
    GATConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]


class GATClassifier(nn.Module):
    """
    Graph Attention Network classifier for LEC trafficking prediction.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Number of output classes
        heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if GATConv is None or global_mean_pool is None:
            raise ImportError("torch-geometric is required for GATClassifier")

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run a forward pass."""
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        if batch is None:
            return self.lin(x)

        x = global_mean_pool(x, batch)
        return self.lin(x)
