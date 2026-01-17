import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch-geometric is required for GNN figures.") from exc

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.explain.config import ModelConfig

    _HAS_NEW_EXPLAINER = True
except Exception:  # pragma: no cover - fallback to legacy API
    try:
        from torch_geometric.nn.models import GNNExplainer

        _HAS_NEW_EXPLAINER = False
    except Exception as exc:  # pragma: no cover
        raise ImportError("GNNExplainer is required for edge attributions.") from exc

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


def set_plot_style() -> None:
    sns.set_theme(style="white", context="paper", palette="pastel")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def build_gene_index(
    grn: pd.DataFrame, adata: sc.AnnData, max_genes: Optional[int]
) -> Tuple[List[str], Dict[str, int]]:
    genes = set(grn["TF"]).union(set(grn["target"]))
    genes = [g for g in genes if g in adata.var_names]

    degree = pd.concat([grn["TF"].value_counts(), grn["target"].value_counts()], axis=1).fillna(0)
    degree["total"] = degree.sum(axis=1)
    degree_map = degree["total"].to_dict()
    genes_sorted = sorted(genes, key=lambda g: degree_map.get(g, 0.0), reverse=True)

    if max_genes:
        keep = set(genes_sorted[:max_genes])
        keep.update({g for g in GLYCO_GENES if g in genes})
        keep.update({g for g in TRAFFICKING_GENES if g in genes})
        genes_sorted = [g for g in genes_sorted if g in keep]

    gene_to_idx = {g: i for i, g in enumerate(genes_sorted)}
    return genes_sorted, gene_to_idx


def build_edges(
    grn: pd.DataFrame, gene_to_idx: Dict[str, int], min_importance: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    edges = grn[grn["importance"] >= min_importance].copy()
    edges = edges[edges["TF"].isin(gene_to_idx) & edges["target"].isin(gene_to_idx)]
    edge_index = torch.tensor(
        [[gene_to_idx[tf] for tf in edges["TF"]], [gene_to_idx[tgt] for tgt in edges["target"]]],
        dtype=torch.long,
    )
    edge_attr = torch.tensor(edges["importance"].values, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def build_node_types(
    gene_list: List[str], tf_set: set[str]
) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[int]]:
    node_types = np.full(len(gene_list), 3)
    tf_idx: List[int] = []
    glyco_idx: List[int] = []
    traffic_idx: List[int] = []
    for idx, gene in enumerate(gene_list):
        if gene in tf_set:
            node_types[idx] = 0
            tf_idx.append(idx)
        if gene in GLYCO_GENES:
            node_types[idx] = 1
            glyco_idx.append(idx)
        if gene in TRAFFICKING_GENES:
            node_types[idx] = 2
            traffic_idx.append(idx)
    node_types_tensor = torch.tensor(node_types, dtype=torch.long)

    one_hot = np.zeros((len(gene_list), 3), dtype=np.float32)
    for idx, node_type in enumerate(node_types):
        if node_type in (0, 1, 2):
            one_hot[idx, node_type] = 1.0
    return node_types_tensor, torch.tensor(one_hot, dtype=torch.float32), tf_idx, traffic_idx


def compute_labels(
    adata: sc.AnnData, label_quantile: float, trim_middle: bool
) -> Tuple[np.ndarray, np.ndarray]:
    traffic_genes = [g for g in TRAFFICKING_GENES if g in adata.var_names]
    if not traffic_genes:
        raise ValueError("No trafficking genes found for labels.")
    expr = adata[:, traffic_genes].X
    if hasattr(expr, "toarray"):
        expr = expr.toarray()
    score = np.asarray(expr).mean(axis=1)
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


class CellGraphDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        expr_matrix,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_types: torch.Tensor,
        node_type_features: torch.Tensor,
        tf_idx: List[int],
        exclude_tf: bool,
    ) -> None:
        self.expr_matrix = expr_matrix
        self.edge_index = edge_index.cpu()
        self.edge_attr = edge_attr.cpu()
        self.node_types = node_types.cpu()
        self.node_type_features = node_type_features.cpu()
        self.tf_idx = tf_idx
        self.exclude_tf = exclude_tf

    def __len__(self) -> int:
        return self.expr_matrix.shape[0]

    def __getitem__(self, idx: int) -> Data:
        row = self.expr_matrix[idx]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()
        x = np.asarray(row, dtype=np.float32).reshape(-1, 1)
        if self.exclude_tf and self.tf_idx:
            x[self.tf_idx, :] = 0.0
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.cat([x, self.node_type_features], dim=1)
        return Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            node_types=self.node_types,
        )


def predict_scores(
    model: LecGNN,
    expr_matrix,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: torch.Tensor,
    node_type_features: torch.Tensor,
    tf_idx: List[int],
    device: torch.device,
    batch_size: int,
    exclude_tf: bool,
) -> np.ndarray:
    dataset = CellGraphDataset(
        expr_matrix,
        edge_index,
        edge_attr,
        node_types,
        node_type_features,
        tf_idx,
        exclude_tf=exclude_tf,
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.node_types, batch.edge_attr, batch.batch)
            logits.append(output.cpu())
    logits = torch.cat(logits).numpy()
    return 1 / (1 + np.exp(-logits))


def ensure_umap(adata: sc.AnnData, seed: int) -> None:
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, svd_solver="arpack")
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, random_state=seed)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=seed)


def plot_umap_predictions(
    adata: sc.AnnData,
    score_col: str,
    label_col: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = {"Low trafficking": "#BFD7EA", "High trafficking": "#F4B6C2"}
    sc.pl.umap(
        adata,
        color=label_col,
        palette=palette,
        ax=ax,
        show=False,
        frameon=False,
        size=8,
        title="GNN predictions (high/low trafficking)",
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_ablation_barplot(results: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(
        data=results,
        x="ablation",
        y="auc",
        palette="pastel",
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC")
    ax.set_xlabel("")
    ax.set_title("Ablation comparison (AUC)")
    for idx, row in results.iterrows():
        ax.text(idx, row["auc"] + 0.02, f"{row['auc']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_edge_attributions(attrib_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=attrib_df,
        y="edge",
        x="attribution",
        palette="pastel",
        ax=ax,
    )
    ax.set_xlabel("Attribution score")
    ax.set_ylabel("")
    ax.set_title("Top 10 edge attributions (GNNExplainer)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_gnn_explainer(
    model: LecGNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_types: torch.Tensor,
) -> torch.Tensor:
    if _HAS_NEW_EXPLAINER:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=ModelConfig(mode="binary_classification", task_level="graph", return_type="raw"),
        )
        explanation = explainer(
            x,
            edge_index,
            edge_attr=edge_attr,
            node_types=node_types,
        )
        return explanation.edge_mask

    class _Wrapper(nn.Module):
        def __init__(self, base: LecGNN, node_types: torch.Tensor, edge_attr: torch.Tensor) -> None:
            super().__init__()
            self.base = base
            self.node_types = node_types
            self.edge_attr = edge_attr

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            return self.base(x, edge_index, self.node_types, self.edge_attr, batch=None)

    wrapper = _Wrapper(model, node_types, edge_attr)
    explainer = GNNExplainer(wrapper, epochs=200)
    _, edge_mask = explainer.explain_graph(x, edge_index)
    return edge_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GNN figures (UMAP, ablations, attributions)")
    parser.add_argument("--adata", default="data/processed/GSE282417_processed.h5ad")
    parser.add_argument("--grn", default="results/grn/GSE282417_grn_small.csv")
    parser.add_argument("--checkpoint", default="models/lec_gnn_checkpoint.pt")
    parser.add_argument("--output-dir", default="results/figures/gnn")
    parser.add_argument("--max-genes", type=int, default=800)
    parser.add_argument("--min-importance", type=float, default=0.10)
    parser.add_argument("--sample-cells", type=int, default=8000)
    parser.add_argument("--label-quantile", type=float, default=0.8)
    parser.add_argument("--trim-middle", action="store_true")
    parser.add_argument("--exclude-trafficking-features", action="store_true", default=True)
    parser.add_argument("--pred-quantile", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--explain-index", type=int, default=None)
    args = parser.parse_args()

    set_plot_style()
    set_torch_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading AnnData from %s", args.adata)
    adata = sc.read_h5ad(args.adata)
    logger.info("Loading GRN from %s", args.grn)
    grn = pd.read_csv(args.grn)

    if args.sample_cells and args.sample_cells < adata.n_obs:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(adata.n_obs, size=args.sample_cells, replace=False)
        adata = adata[idx].copy()
        logger.info("Sampled %d cells for figures", adata.n_obs)

    gene_list, gene_to_idx = build_gene_index(grn, adata, args.max_genes)
    edge_index, edge_attr = build_edges(grn, gene_to_idx, args.min_importance)
    node_types, node_type_features, tf_idx, traffic_idx = build_node_types(gene_list, set(grn["TF"]))

    expr = adata[:, gene_list].X
    if args.exclude_trafficking_features and traffic_idx:
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        expr = np.asarray(expr)
        expr[:, traffic_idx] = 0.0

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = LecGNN(
        in_channels=1 + node_type_features.shape[1],
        hidden_channels=args.hidden_dim,
        num_gat_layers=args.gat_layers,
        heads=args.heads,
        dropout=args.dropout,
        use_glyco_readout=True,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    probs = predict_scores(
        model,
        expr,
        edge_index,
        edge_attr,
        node_types,
        node_type_features,
        tf_idx,
        device,
        args.batch_size,
        exclude_tf=False,
    )
    adata.obs["gnn_pred"] = probs
    thresh = np.quantile(probs, args.pred_quantile)
    adata.obs["gnn_pred_label"] = pd.Categorical(
        np.where(probs >= thresh, "High trafficking", "Low trafficking"),
        categories=["Low trafficking", "High trafficking"],
    )

    ensure_umap(adata, args.seed)
    plot_umap_predictions(
        adata,
        "gnn_pred",
        "gnn_pred_label",
        output_dir / "figC_umap_gnn_predictions.png",
    )

    labels, keep_mask = compute_labels(adata, args.label_quantile, args.trim_middle)
    eval_expr = expr
    if args.trim_middle:
        eval_expr = eval_expr[keep_mask]
        labels = labels[keep_mask]

    def auc_for(edge_idx: torch.Tensor, edge_att: torch.Tensor, name: str, zero_tf: bool = False) -> float:
        pred = predict_scores(
            model,
            eval_expr,
            edge_idx,
            edge_att,
            node_types,
            node_type_features,
            tf_idx,
            device,
            args.batch_size,
            exclude_tf=zero_tf,
        )
        return roc_auc_score(labels, pred)

    rng = np.random.default_rng(args.seed)
    shuffled_targets = edge_index[1].clone()
    shuffled_targets = torch.tensor(rng.permutation(shuffled_targets.numpy()), dtype=torch.long)
    random_edge_index = torch.stack([edge_index[0], shuffled_targets])

    glyco_targets = [gene_to_idx[g] for g in GLYCO_GENES if g in gene_to_idx]
    glyco_mask = np.isin(edge_index[1].numpy(), glyco_targets)
    no_glyco_edge_index = edge_index[:, ~glyco_mask]
    no_glyco_edge_attr = edge_attr[~glyco_mask]

    ablation_results = pd.DataFrame(
        [
            {"ablation": "Baseline", "auc": auc_for(edge_index, edge_attr, "baseline")},
            {"ablation": "Random graph", "auc": auc_for(random_edge_index, edge_attr, "random")},
            {"ablation": "No glyco edges", "auc": auc_for(no_glyco_edge_index, no_glyco_edge_attr, "no_glyco")},
            {"ablation": "TF knockout", "auc": auc_for(edge_index, edge_attr, "tf_knockout", zero_tf=True)},
        ]
    )
    plot_ablation_barplot(ablation_results, output_dir / "figD_ablation_auc.png")

    explain_idx = int(args.explain_index) if args.explain_index is not None else int(np.argmax(probs))
    row = expr[explain_idx]
    if hasattr(row, "toarray"):
        row = row.toarray().ravel()
    x = torch.tensor(np.asarray(row, dtype=np.float32).reshape(-1, 1), device=device)
    x = torch.cat([x, node_type_features.to(device)], dim=1)
    edge_mask = run_gnn_explainer(
        model,
        x,
        edge_index.to(device),
        edge_attr.to(device),
        node_types.to(device),
    )
    edge_mask = edge_mask.detach().cpu().numpy()
    top_idx = np.argsort(edge_mask)[-10:][::-1]
    edge_labels = [
        f"{gene_list[edge_index[0, i]]}→{gene_list[edge_index[1, i]]}" for i in top_idx
    ]
    attrib_df = pd.DataFrame({"edge": edge_labels, "attribution": edge_mask[top_idx]})
    plot_edge_attributions(attrib_df, output_dir / "figE_edge_attributions.png")

    logger.info("Saved figures to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
