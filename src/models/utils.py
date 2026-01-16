import random
from typing import Dict

import numpy as np
import torch
from torch import nn


def set_torch_seed(seed: int) -> None:
    """Set random seeds for reproducibility in PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction and KL divergence loss for a VAE."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Train a model for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(*batch) if isinstance(batch, (list, tuple)) else model(batch)
        if isinstance(outputs, tuple) and len(outputs) == 3:
            loss = vae_loss(outputs[0], batch[0] if isinstance(batch, (list, tuple)) else batch, outputs[1], outputs[2])
        else:
            loss = outputs
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
    """Evaluate a model and return summary metrics."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(*batch) if isinstance(batch, (list, tuple)) else model(batch)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                loss = vae_loss(outputs[0], batch[0] if isinstance(batch, (list, tuple)) else batch, outputs[1], outputs[2])
            else:
                loss = outputs
            total_loss += float(loss.item())
    return {"loss": total_loss / max(len(loader), 1)}
