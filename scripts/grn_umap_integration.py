import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc
import seaborn as sns

logger = logging.getLogger(__name__)

GLYCO_GENES = [
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
]

TRAFFICKING_GENES = [
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
]

PASTEL_COLORS = {
    "tf": "#FBB4AE",
    "glyco": "#B3CDE3",
    "traffic": "#CCEBC5",
    "feature": "#DECBE4",
}


def set_plot_style() -> None:
    sns.set_theme(style="white", context="paper", palette="pastel")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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


def pastel_cmap(color: str) -> matplotlib.colors.Colormap:
    return sns.light_palette(color, as_cmap=True)


def safe_score_genes(adata: sc.AnnData, genes: list[str], score_name: str) -> None:
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 2:
        logger.warning("Skipping score %s: only %d genes present.", score_name, len(present))
        return
    sc.tl.score_genes(adata, present, score_name=score_name, use_raw=False)
    logger.info("Scored %s with %d genes.", score_name, len(present))


def ensure_embeddings(adata: sc.AnnData, n_neighbors: int, seed: int) -> None:
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, svd_solver="arpack")
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=seed)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, random_state=seed)


def plot_embedding_grid(
    adata: sc.AnnData,
    basis: str,
    color_keys: list[str],
    output_path: Path,
    cmaps: dict[str, matplotlib.colors.Colormap],
    point_size: float,
) -> None:
    ncols = 2
    nrows = int(np.ceil(len(color_keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = np.atleast_1d(axes).ravel()

    for idx, key in enumerate(color_keys):
        ax = axes[idx]
        if key not in adata.obs and key not in adata.var_names:
            ax.set_axis_off()
            continue
        if key in adata.obs and pd.api.types.is_categorical_dtype(adata.obs[key]):
            sc.pl.embedding(
                adata,
                basis=basis,
                color=key,
                ax=ax,
                show=False,
                frameon=False,
                size=point_size,
                legend_loc="on data",
                palette="pastel",
            )
        else:
            sc.pl.embedding(
                adata,
                basis=basis,
                color=key,
                ax=ax,
                show=False,
                frameon=False,
                size=point_size,
                color_map=cmaps.get(key, pastel_cmap(PASTEL_COLORS["feature"])),
            )
        ax.set_title(key.replace("_", " "), fontsize=11, fontweight="bold")

    for idx in range(len(color_keys), len(axes)):
        axes[idx].set_axis_off()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP/PCA integration plots for GRN modules")
    parser.add_argument(
        "--adata",
        default="data/processed/GSE282417_processed.h5ad",
        help="Path to AnnData file",
    )
    parser.add_argument(
        "--grn",
        default="results/grn/GSE282417_grn_small.csv",
        help="Path to GRN CSV with TF,target,importance",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures/grn",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--tf-modules",
        default="IRF1,NFKB1,PROX1,STAT3",
        help="Comma-separated TFs to score as modules",
    )
    parser.add_argument(
        "--module-size",
        type=int,
        default=15,
        help="Top targets per TF for module score",
    )
    parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors for UMAP")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--point-size", type=float, default=8, help="Point size")
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA panel")
    parser.add_argument("--skip-umap", action="store_true", help="Skip UMAP panel")
    parser.add_argument("--cluster-key", default="leiden", help="Cluster key for optional panel")
    args = parser.parse_args()

    set_plot_style()
    adata_path = Path(args.adata)
    grn_path = Path(args.grn)
    output_dir = Path(args.output_dir)

    logger.info("Loading AnnData from %s", adata_path)
    adata = sc.read_h5ad(adata_path)
    logger.info("Loading GRN from %s", grn_path)
    grn = pd.read_csv(grn_path)

    required = {"TF", "target", "importance"}
    if not required.issubset(grn.columns):
        missing = required - set(grn.columns)
        raise ValueError(f"GRN file missing columns: {sorted(missing)}")

    tf_list = [tf.strip() for tf in args.tf_modules.split(",") if tf.strip()]
    for tf in tf_list:
        targets = (
            grn.loc[grn["TF"] == tf]
            .nlargest(args.module_size, "importance")["target"]
            .tolist()
        )
        safe_score_genes(adata, targets, f"{tf}_module")

    safe_score_genes(adata, GLYCO_GENES, "glyco_score")
    safe_score_genes(adata, TRAFFICKING_GENES, "trafficking_score")

    ensure_embeddings(adata, n_neighbors=args.n_neighbors, seed=args.seed)

    module_keys = [f"{tf}_module" for tf in tf_list]
    score_keys = ["glyco_score", "trafficking_score"]
    color_keys = module_keys + score_keys

    cmaps = {
        "glyco_score": pastel_cmap(PASTEL_COLORS["glyco"]),
        "trafficking_score": pastel_cmap(PASTEL_COLORS["traffic"]),
    }
    for tf in tf_list:
        cmaps[f"{tf}_module"] = pastel_cmap(PASTEL_COLORS["tf"])

    if not args.skip_umap:
        plot_embedding_grid(
            adata,
            basis="umap",
            color_keys=color_keys[:4],
            output_path=output_dir / "fig6_umap_grn_modules",
            cmaps=cmaps,
            point_size=args.point_size,
        )

    if not args.skip_pca:
        plot_embedding_grid(
            adata,
            basis="pca",
            color_keys=color_keys[:4],
            output_path=output_dir / "fig7_pca_grn_modules",
            cmaps=cmaps,
            point_size=args.point_size,
        )

    feature_keys = ["IRF1", "CCL19", "ST6GALNAC3", "VCAM1"]
    plot_embedding_grid(
        adata,
        basis="umap",
        color_keys=feature_keys,
        output_path=output_dir / "fig8_umap_feature_validation",
        cmaps={key: pastel_cmap(PASTEL_COLORS["feature"]) for key in feature_keys},
        point_size=args.point_size,
    )

    if args.cluster_key in adata.obs:
        plot_embedding_grid(
            adata,
            basis="umap",
            color_keys=[args.cluster_key],
            output_path=output_dir / "fig9_umap_clusters",
            cmaps=cmaps,
            point_size=args.point_size,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
