import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("torch-geometric is required for training.") from exc

from src.models.lec_gnn import LecGNN
from src.models.utils import set_torch_seed

logger = logging.getLogger(__name__)

GLYCO_GENES = {
    "ST3GAL1",
    "ST3GAL2",
    "ST3GAL3",
    "ST3GAL4",
    "ST3GAL5",
    "ST3GAL6",
    "ST6GAL1",
    "ST6GAL2",
    "ST6GALNAC1",
    "ST6GALNAC2",
    "ST6GALNAC3",
    "ST6GALNAC4",
    "ST6GALNAC5",
    "ST6GALNAC6",
    "ST8SIA1",
    "ST8SIA4",
    "GALNT1",
    "GALNT2",
    "GALNT3",
    "GALNT4",
    "GALNT5",
    "GALNT6",
    "GALNT7",
    "GALNT10",
    "GALNT11",
    "GALNT12",
    "GALNT14",
    "B3GALT1",
    "B3GALT2",
    "B3GALT4",
    "B3GALT6",
    "B4GALT1",
    "B4GALT2",
    "B4GALT3",
    "B4GALT4",
    "B4GALT5",
    "B4GALT6",
    "MGAT1",
    "MGAT2",
    "MGAT3",
    "MGAT4A",
    "MGAT4B",
    "MGAT5",
    "MGAT5B",
    "FUT1",
    "FUT2",
    "FUT3",
    "FUT4",
    "FUT6",
    "FUT8",
    "FUT9",
    "A3GALT2",
    "B3GNT2",
    "B3GNT3",
    "B3GNT7",
    "B3GNT8",
    "GCNT1",
    "GCNT2",
    "GCNT3",
    "GCNT4",
    "C1GALT1",
    "C1GALT1C1",
    "NEU1",
    "NEU3",
    "FUCA1",
    "FUCA2",
    "HEXA",
    "HEXB",
    "SLC35A1",
    "SLC35A2",
    "SLC35B1",
    "SLC35C1",
    "SLC35D1",
}

TRAFFICKING_GENES = {
    "VCAM1",
    "ICAM1",
    "ICAM2",
    "ICAM3",
    "MADCAM1",
    "ALCAM",
    "PECAM1",
    "CD34",
    "SELE",
    "SELP",
    "SELL",
    "CCL19",
    "CCL21",
    "CCL2",
    "CCL5",
    "CXCL9",
    "CXCL10",
    "CXCL11",
    "CXCL12",
    "CXCL13",
    "CX3CL1",
    "CCR7",
    "CXCR4",
    "CXCR5",
    "ACKR2",
    "ACKR4",
    "LYVE1",
    "PDPN",
    "FLT4",
    "NRP2",
    "VEGFC",
    "VEGFD",
    "CD274",
    "PDCD1LG2",
    "CD80",
    "CD86",
    "CTLA4",
    "PLPP3",
    "SPHK1",
    "S1PR1",
    "S1PR2",
    "F3",
    "TNFRSF1A",
    "TNFRSF1B",
    "CD53",
    "CD44",
    "ITGA4",
    "ITGAL",
    "ITGAM",
    "ITGB1",
    "ITGB2",
    "ITGB7",
}


class CellGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        expr_matrix,
        labels: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type_features: torch.Tensor,
        node_types: torch.Tensor,
    ) -> None:
        self.expr_matrix = expr_matrix
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_type_features = node_type_features
        self.node_types = node_types

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Data:
        row = self.expr_matrix[idx]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()
        x = torch.tensor(row, dtype=torch.float32).unsqueeze(1)
        if self.node_type_features is not None:
            x = torch.cat([x, self.node_type_features], dim=1)
        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.labels[idx],
            node_types=self.node_types,
        )
        return data


def build_gene_index(
    grn: pd.DataFrame,
    adata: sc.AnnData,
    max_genes: Optional[int],
) -> Tuple[List[str], Dict[str, int]]:
    genes = set(grn["TF"]).union(set(grn["target"]))
    genes = [g for g in genes if g in adata.var_names]

    degree = pd.concat(
        [grn["TF"].value_counts(), grn["target"].value_counts()], axis=1
    ).fillna(0)
    degree["total"] = degree.sum(axis=1)
    degree = degree["total"].to_dict()
    genes_sorted = sorted(genes, key=lambda g: degree.get(g, 0.0), reverse=True)

    if max_genes:
        keep = set(genes_sorted[:max_genes])
        keep.update({g for g in GLYCO_GENES if g in genes})
        keep.update({g for g in TRAFFICKING_GENES if g in genes})
        genes_sorted = [g for g in genes_sorted if g in keep]

    gene_to_idx = {g: i for i, g in enumerate(genes_sorted)}
    return genes_sorted, gene_to_idx


def build_edges(
    grn: pd.DataFrame,
    gene_to_idx: Dict[str, int],
    min_importance: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = grn.copy()
    if min_importance is not None:
        edges = edges[edges["importance"] >= min_importance]
    edges = edges[edges["TF"].isin(gene_to_idx) & edges["target"].isin(gene_to_idx)]

    edge_index = torch.tensor(
        [[gene_to_idx[tf] for tf in edges["TF"]], [gene_to_idx[tgt] for tgt in edges["target"]]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(edges["importance"].values, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def compute_labels(
    adata: sc.AnnData,
    gene_list: List[str],
    label_quantile: float,
    trim_middle: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    traffic_genes = [g for g in TRAFFICKING_GENES if g in adata.var_names]
    if not traffic_genes:
        raise ValueError("No trafficking genes found in AnnData.")
    traffic_expr = adata[:, traffic_genes].X
    if hasattr(traffic_expr, "toarray"):
        traffic_expr = traffic_expr.toarray()
    score = np.asarray(traffic_expr).mean(axis=1)

    if not trim_middle:
        threshold = np.quantile(score, label_quantile)
        labels = (score >= threshold).astype(int)
        keep = np.ones_like(labels, dtype=bool)
        return labels, keep

    low = np.quantile(score, 1.0 - label_quantile)
    high = np.quantile(score, label_quantile)
    keep = (score <= low) | (score >= high)
    labels = (score >= high).astype(int)
    return labels, keep


def build_node_types(gene_list: List[str], tf_set: set[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    node_types = np.full(len(gene_list), 3)
    for idx, gene in enumerate(gene_list):
        if gene in tf_set:
            node_types[idx] = 0
        if gene in GLYCO_GENES:
            node_types[idx] = 1
        if gene in TRAFFICKING_GENES:
            node_types[idx] = 2
    node_types_tensor = torch.tensor(node_types, dtype=torch.long)

    one_hot = np.zeros((len(gene_list), 3), dtype=np.float32)
    for idx, node_type in enumerate(node_types):
        if node_type in (0, 1, 2):
            one_hot[idx, node_type] = 1.0
    return node_types_tensor, torch.tensor(one_hot, dtype=torch.float32)


def train_epoch(
    model: LecGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.node_types, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: LecGNN,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.node_types, batch.edge_attr, batch.batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
            if criterion is not None:
                losses.append(float(criterion(logits, batch.y).item()))
    if not all_logits:
        return {"auc": 0.0, "auprc": 0.0, "loss": 0.0}
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    metrics = {
        "auc": roc_auc_score(labels, probs),
        "auprc": average_precision_score(labels, probs),
    }
    if criterion is not None:
        metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def save_training_plots(history: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train_loss", color="#A3C4DC")
    ax.plot(history["epoch"], history["val_loss"], label="val_loss", color="#F7C59F")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training/Validation Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "train_val_loss.png", dpi=300)
    fig.savefig(output_dir / "train_val_loss.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["epoch"], history["val_auc"], label="val_auc", color="#B7E4C7")
    ax.plot(history["epoch"], history["val_auprc"], label="val_auprc", color="#F2B5D4")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation AUC/AUPRC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "val_auc_auprc.png", dpi=300)
    fig.savefig(output_dir / "val_auc_auprc.pdf")
    plt.close(fig)


def save_roc_pr_curves(labels: np.ndarray, probs: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#9DB4C0", lw=2, label=f"AUC = {roc_auc_score(labels, probs):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#D3D3D3", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=300)
    fig.savefig(output_dir / "roc_curve.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#F2B5D4", lw=2, label=f"AUPRC = {average_precision_score(labels, probs):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "pr_curve.png", dpi=300)
    fig.savefig(output_dir / "pr_curve.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lec-GNN for trafficking prediction")
    parser.add_argument("--adata", default="data/processed/GSE282417_processed.h5ad")
    parser.add_argument("--grn", default="results/grn/GSE282417_grn_small.csv")
    parser.add_argument("--output", default="models/lec_gnn_checkpoint.pt")
    parser.add_argument("--max-genes", type=int, default=800)
    parser.add_argument("--min-importance", type=float, default=0.0)
    parser.add_argument("--sample-cells", type=int, default=8000)
    parser.add_argument("--label-quantile", type=float, default=0.8)
    parser.add_argument("--trim-middle", action="store_true", help="Keep only top/bottom quantiles")
    parser.add_argument("--exclude-trafficking-features", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--plot-dir", default="results/gnn/plots")
    parser.add_argument("--metrics-csv", default="results/gnn/training_metrics.csv")
    args = parser.parse_args()

    set_torch_seed(args.seed)

    adata_path = Path(args.adata)
    grn_path = Path(args.grn)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AnnData from %s", adata_path)
    adata = sc.read_h5ad(adata_path)
    logger.info("Loading GRN from %s", grn_path)
    grn = pd.read_csv(grn_path)

    if args.sample_cells and args.sample_cells < adata.n_obs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.n_obs, size=args.sample_cells, replace=False)
        adata = adata[idx].copy()
        logger.info("Sampled %d cells", adata.n_obs)

    gene_list, gene_to_idx = build_gene_index(grn, adata, args.max_genes)
    edge_index, edge_attr = build_edges(grn, gene_to_idx, args.min_importance)

    labels, keep_mask = compute_labels(adata, gene_list, args.label_quantile, args.trim_middle)
    if args.trim_middle:
        adata = adata[keep_mask].copy()
        labels = labels[keep_mask]
        logger.info("Trimmed to %d cells for labels", adata.n_obs)

    expr = adata[:, gene_list].X
    if args.exclude_trafficking_features:
        traffic_idx = [gene_to_idx[g] for g in TRAFFICKING_GENES if g in gene_to_idx]
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr)
        expr[:, traffic_idx] = 0.0
    node_types, node_type_features = build_node_types(gene_list, set(grn["TF"]))

    x_train, x_temp, y_train, y_temp = train_test_split(
        np.arange(adata.n_obs), labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    def subset_expr(indices: np.ndarray):
        return expr[indices]

    train_ds = CellGraphDataset(subset_expr(x_train), y_train, edge_index, edge_attr, node_type_features, node_types)
    val_ds = CellGraphDataset(subset_expr(x_val), y_val, edge_index, edge_attr, node_type_features, node_types)
    test_ds = CellGraphDataset(subset_expr(x_test), y_test, edge_index, edge_attr, node_type_features, node_types)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = LecGNN(
        in_channels=1 + node_type_features.shape[1],
        hidden_channels=args.hidden_dim,
        num_gat_layers=args.gat_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_glyco_readout=True,
    ).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    best_state = None
    patience = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device, criterion)
        logger.info(
            "Epoch %d | loss=%.4f | val_loss=%.4f | val_auc=%.3f | val_auprc=%.3f",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["auc"],
            val_metrics["auprc"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "val_auprc": val_metrics["auprc"],
            }
        )
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = model.state_dict()
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_path)
        logger.info("Saved best model to %s", output_path)

    test_metrics = evaluate(model, test_loader, device, criterion)
    logger.info("Test AUC: %.3f | Test AUPRC: %.3f", test_metrics["auc"], test_metrics["auprc"])

    history_df = pd.DataFrame(history)
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(metrics_path, index=False)
    logger.info("Saved training history to %s", metrics_path)

    plot_dir = Path(args.plot_dir)
    save_training_plots(history_df, plot_dir)

    # Recompute probabilities for ROC/PR curves on test set.
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.node_types, batch.edge_attr, batch.batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch.y.cpu())
    if all_logits:
        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs = 1 / (1 + np.exp(-logits))
        save_roc_pr_curves(labels, probs, plot_dir)
        logger.info("Saved ROC/PR curves to %s", plot_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
