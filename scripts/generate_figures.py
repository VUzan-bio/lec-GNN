import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.patches import Patch

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

TF_FAMILIES = {
    "NF-kB": ["NFKB1", "NFKB2", "RELA", "RELB", "REL"],
    "ETS": ["ELK3", "ERG", "ETS1", "ETS2", "FLI1", "ETV2"],
    "LEC": ["PROX1", "SOX18", "SOX7", "SOX17", "FOXC1", "FOXC2"],
    "Interferon": ["IRF1", "IRF3", "IRF7"],
    "STAT": ["STAT1", "STAT3", "STAT5A", "STAT5B"],
    "Hypoxia": ["HIF1A", "EPAS1", "ARNT"],
    "KLF": ["KLF2", "KLF4"],
    "Other": ["GATA2", "GATA3", "GATA4", "TAL1"],
}

FAMILY_COLORS = {
    "NF-kB": "#FBB4AE",
    "ETS": "#B3CDE3",
    "LEC": "#CCEBC5",
    "Interferon": "#DECBE4",
    "STAT": "#FED9A6",
    "Hypoxia": "#FFFFCC",
    "KLF": "#E5D8BD",
    "Other": "#F2F2F2",
}

TARGET_COLORS = {"glyco": "#B3CDE3", "trafficking": "#FBB4AE", "other": "#F2F2F2"}

TARGET_CLASS_ORDER = [
    "Sialyltransferases",
    "GalNAc transferases",
    "Galactosyltransferases",
    "Fucosyltransferases",
    "N-glycan enzymes",
    "Other glyco",
    "Chemokines",
    "Adhesion molecules",
    "Integrins",
    "Selectins",
    "Immune checkpoints",
    "LEC markers",
    "Other trafficking",
]

TARGET_CLASS_COLORS = {
    "Sialyltransferases": "#B3CDE3",
    "GalNAc transferases": "#CCEBC5",
    "Galactosyltransferases": "#DECBE4",
    "Fucosyltransferases": "#FED9A6",
    "N-glycan enzymes": "#FFFFCC",
    "Other glyco": "#E5D8BD",
    "Chemokines": "#FBB4AE",
    "Adhesion molecules": "#FDDAEC",
    "Integrins": "#CBD5E8",
    "Selectins": "#F1E2CC",
    "Immune checkpoints": "#F4CAE4",
    "LEC markers": "#B3E2CD",
    "Other trafficking": "#F2F2F2",
}


def set_plot_style() -> None:
    sns.set_theme(style="white", context="paper", palette="pastel")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#E6E6E6",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
        }
    )


def resolve_grn_path(value: Optional[str]) -> Path:
    if value:
        return Path(value)
    candidates = [
        Path("results/grn/GSE282417_grn_comprehensive.csv"),
        Path("results/grn/GSE282417_grn_small.csv"),
        Path("results/grn/GSE282417_grn_comprehensive_10k.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No GRN file found. Pass --grn with a CSV that has TF,target,importance columns."
    )


def classify_tf_family(gene: str) -> str:
    for family, members in TF_FAMILIES.items():
        if gene in members:
            return family
    return "Other"


def classify_target(gene: str) -> str:
    if gene in GLYCO_GENES:
        if gene.startswith(("ST3GAL", "ST6GAL", "ST8SIA")):
            return "Sialyltransferases"
        if gene.startswith("GALNT"):
            return "GalNAc transferases"
        if gene.startswith(("B3GALT", "B4GALT")):
            return "Galactosyltransferases"
        if gene.startswith("FUT"):
            return "Fucosyltransferases"
        if gene.startswith("MGAT"):
            return "N-glycan enzymes"
        return "Other glyco"
    if gene in TRAFFICKING_GENES:
        if gene.startswith(("CCL", "CXCL", "CX3CL")):
            return "Chemokines"
        if gene.startswith(("ICAM", "VCAM", "PECAM", "ALCAM", "MADCAM")):
            return "Adhesion molecules"
        if gene.startswith(("CD", "PDCD", "CTLA")):
            return "Immune checkpoints"
        if gene in {"LYVE1", "PDPN", "FLT4", "NRP2", "VEGFC", "VEGFD"}:
            return "LEC markers"
        if gene.startswith("ITG"):
            return "Integrins"
        if gene in {"SELE", "SELP", "SELL"}:
            return "Selectins"
        return "Other trafficking"
    return "Other"


def save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_network_panel(
    ax: plt.Axes,
    edges: pd.DataFrame,
    min_label_degree: int,
) -> None:
    if edges.empty:
        ax.set_axis_off()
        return

    G = nx.from_pandas_edgelist(
        edges, source="TF", target="target", edge_attr="importance", create_using=nx.DiGraph()
    )

    tf_nodes = sorted(
        [n for n in G.nodes() if n not in GLYCO_GENES + TRAFFICKING_GENES],
        key=lambda n: G.degree(n),
        reverse=True,
    )
    glyco_nodes = sorted(
        [n for n in G.nodes() if n in GLYCO_GENES],
        key=lambda n: G.degree(n),
        reverse=True,
    )
    traffic_nodes = sorted(
        [n for n in G.nodes() if n in TRAFFICKING_GENES],
        key=lambda n: G.degree(n),
        reverse=True,
    )

    pos: Dict[str, tuple] = {}
    for nodes, x in [(tf_nodes, 0.0), (glyco_nodes, 0.5), (traffic_nodes, 1.0)]:
        y_positions = np.linspace(0.95, 0.05, max(len(nodes), 1))
        for node, y in zip(nodes, y_positions):
            pos[node] = (x, y)

    node_colors = []
    for node in G.nodes():
        if node in tf_nodes:
            node_colors.append(FAMILY_COLORS[classify_tf_family(node)])
        elif node in glyco_nodes:
            node_colors.append(TARGET_COLORS["glyco"])
        elif node in traffic_nodes:
            node_colors.append(TARGET_COLORS["trafficking"])
        else:
            node_colors.append(TARGET_COLORS["other"])

    weights = np.array([G.edges[e]["importance"] for e in G.edges()])
    w_min, w_max = weights.min(), weights.max()
    if w_max == w_min:
        widths = np.full_like(weights, 1.3)
        alphas = np.full_like(weights, 0.7)
    else:
        widths = 0.6 + 2.6 * (weights - w_min) / (w_max - w_min)
        alphas = 0.25 + 0.75 * (weights - w_min) / (w_max - w_min)

    edge_colors = []
    for u, v in G.edges():
        if v in glyco_nodes:
            edge_colors.append(TARGET_COLORS["glyco"])
        elif v in traffic_nodes:
            edge_colors.append(TARGET_COLORS["trafficking"])
        else:
            edge_colors.append("#C9C9C9")

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=widths,
        alpha=alphas,
        arrows=True,
        arrowsize=10,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    node_sizes = [(G.in_degree(n) + G.out_degree(n)) * 120 + 240 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        edgecolors="white",
        linewidths=1.1,
        ax=ax,
    )

    must_label = {"NFKB1", "PROX1", "IRF1", "STAT3", "ELK3", "ERG", "SOX18", "FOXC2"}
    label_nodes = {
        n: n
        for n in G.nodes()
        if G.degree(n) >= min_label_degree or n in must_label
    }
    nx.draw_networkx_labels(
        G,
        pos,
        labels=label_nodes,
        font_size=8,
        font_family="DejaVu Sans",
        font_weight="bold",
        ax=ax,
    )

    ax.set_title("Hierarchical GRN: TFs -> Glyco -> Trafficking", fontsize=14, fontweight="bold")
    ax.text(0.0, 1.02, "TFs", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.48, 1.02, "Glyco", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")
    ax.text(0.9, 1.02, "Trafficking", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")

    stats = (
        f"Nodes: {G.number_of_nodes()}\n"
        f"Edges: {G.number_of_edges()}\n"
        f"Median importance: {np.median(weights):.3f}"
    )
    ax.text(
        0.02,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D0D0D0"),
    )
    ax.axis("off")


def build_network_figure(
    grn: pd.DataFrame,
    output_dir: Path,
    top_edges: int,
    min_importance: Optional[float],
    min_label_degree: int,
    dpi: int,
) -> None:
    target_focus = grn[grn["target"].isin(GLYCO_GENES + TRAFFICKING_GENES)].copy()
    if min_importance is not None:
        target_focus = target_focus[target_focus["importance"] >= min_importance]
    if top_edges:
        target_focus = target_focus.nlargest(top_edges, "importance")

    if target_focus.empty:
        logger.warning("No edges available for network figure after filtering.")
        return

    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    build_network_panel(ax, target_focus, min_label_degree)

    legend_elements = [
        Patch(facecolor=TARGET_COLORS["glyco"], label="Glycosylation targets", edgecolor="white"),
        Patch(facecolor=TARGET_COLORS["trafficking"], label="Trafficking targets", edgecolor="white"),
    ]
    for family in ["NF-kB", "ETS", "LEC", "Interferon", "STAT", "Hypoxia", "KLF", "Other"]:
        legend_elements.append(
            Patch(facecolor=FAMILY_COLORS[family], label=f"{family} TFs", edgecolor="white")
        )

    fig.legend(handles=legend_elements, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_figure(fig, output_dir, "fig1_network_graph", dpi)


def build_tf_activity_heatmap(
    grn: pd.DataFrame,
    output_dir: Path,
    top_tfs: int,
    dpi: int,
) -> None:
    filtered = grn[grn["target"].isin(GLYCO_GENES + TRAFFICKING_GENES)].copy()
    if filtered.empty:
        logger.warning("No edges available for heatmap figure after filtering.")
        return

    filtered["target_class"] = filtered["target"].apply(classify_target)
    tf_activity = (
        filtered.groupby(["TF", "target_class"])["importance"].sum().reset_index()
    )
    matrix = tf_activity.pivot(index="TF", columns="target_class", values="importance").fillna(0.0)
    matrix["total"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("total", ascending=False).head(top_tfs)
    matrix = matrix.drop(columns=["total"])

    matrix = matrix.reindex(columns=[c for c in TARGET_CLASS_ORDER if c in matrix.columns])

    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = sns.light_palette("#B3CDE3", as_cmap=True)
    display = np.log1p(matrix)
    sns.heatmap(
        display,
        cmap=cmap,
        cbar_kws={"label": "log(1 + total importance)"},
        linewidths=0.6,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Gene class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Transcription factor", fontsize=11, fontweight="bold")
    ax.set_title("TF activity across glyco and trafficking classes", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig2_tf_activity_heatmap", dpi)


def build_tf_class_dotplot(
    grn: pd.DataFrame,
    output_dir: Path,
    top_tfs: int,
    dpi: int,
) -> None:
    filtered = grn[grn["target"].isin(GLYCO_GENES + TRAFFICKING_GENES)].copy()
    if filtered.empty:
        logger.warning("No edges available for TF class dot plot.")
        return

    filtered["target_class"] = filtered["target"].apply(classify_target)
    summary = (
        filtered.groupby(["TF", "target_class"])
        .agg(n_edges=("target", "count"), mean_importance=("importance", "mean"))
        .reset_index()
    )
    tf_totals = summary.groupby("TF")["n_edges"].sum().sort_values(ascending=False).head(top_tfs)
    summary = summary[summary["TF"].isin(tf_totals.index)]

    class_order = [c for c in TARGET_CLASS_ORDER if c in summary["target_class"].unique()]
    tf_order = list(tf_totals.index)
    class_to_x = {name: idx for idx, name in enumerate(class_order)}
    tf_to_y = {name: idx for idx, name in enumerate(tf_order)}

    fig, ax = plt.subplots(figsize=(12, 7))
    sizes = 40 + 260 * (summary["n_edges"] / summary["n_edges"].max())
    scatter = ax.scatter(
        summary["target_class"].map(class_to_x),
        summary["TF"].map(tf_to_y),
        s=sizes,
        c=summary["mean_importance"],
        cmap=sns.light_palette("#B3CDE3", as_cmap=True),
        edgecolors="white",
        linewidths=0.6,
    )

    ax.set_xticks(range(len(class_order)), class_order, rotation=35, ha="right")
    ax.set_yticks(range(len(tf_order)), tf_order)
    ax.set_xlabel("Target class", fontsize=11, fontweight="bold")
    ax.set_ylabel("TF", fontsize=11, fontweight="bold")
    ax.set_title("TF regulatory reach across gene classes", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label("Mean edge importance", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig3_tf_class_dotplot", dpi)


def build_top_regulators_bars(
    grn: pd.DataFrame,
    output_dir: Path,
    top_tfs: int,
    dpi: int,
) -> None:
    glyco_edges = grn[grn["target"].isin(GLYCO_GENES)].copy()
    traffic_edges = grn[grn["target"].isin(TRAFFICKING_GENES)].copy()

    glyco_sum = glyco_edges.groupby("TF")["importance"].sum().sort_values(ascending=False).head(top_tfs)
    traffic_sum = traffic_edges.groupby("TF")["importance"].sum().sort_values(ascending=False).head(top_tfs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    sns.barplot(
        x=glyco_sum.values,
        y=glyco_sum.index,
        color=TARGET_COLORS["glyco"],
        ax=axes[0],
    )
    axes[0].set_title("Top TFs regulating glyco genes", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Total importance")
    axes[0].set_ylabel("TF")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    sns.barplot(
        x=traffic_sum.values,
        y=traffic_sum.index,
        color=TARGET_COLORS["trafficking"],
        ax=axes[1],
    )
    axes[1].set_title("Top TFs regulating trafficking genes", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Total importance")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, output_dir, "fig4_top_regulators", dpi)


def build_importance_distribution(
    grn: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    df = grn.copy()
    df["target_group"] = np.where(
        df["target"].isin(GLYCO_GENES),
        "Glycosylation",
        np.where(df["target"].isin(TRAFFICKING_GENES), "Trafficking", "Other"),
    )
    df = df[df["target_group"].isin(["Glycosylation", "Trafficking"])]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.violinplot(
        data=df,
        x="target_group",
        y="importance",
        palette=[TARGET_COLORS["glyco"], TARGET_COLORS["trafficking"]],
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Edge importance", fontsize=11, fontweight="bold")
    ax.set_title("Importance distribution by target class", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig5_importance_distribution", dpi)


def write_legend_file(output_dir: Path, grn_path: Path) -> None:
    legends = [
        "Figure 1: Hierarchical GRN with TFs -> glyco -> trafficking, colored by TF family and target class.",
        "Figure 2: Heatmap of TF regulatory activity across glyco and trafficking gene classes (log-scaled).",
        "Figure 3: Dot plot of TF reach across gene classes (dot size = edge count, color = mean importance).",
        "Figure 4: Top TFs for glyco vs trafficking targets (total importance).",
        "Figure 5: Importance distribution split by target class.",
        f"GRN source: {grn_path.as_posix()}",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figure_legends.txt").write_text("\n".join(legends), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--grn", default=None, help="Path to GRN CSV with TF,target,importance")
    parser.add_argument("--output", default="results/figures", help="Output directory")
    parser.add_argument("--top-edges", type=int, default=200, help="Top edges for network figure")
    parser.add_argument("--min-importance", type=float, default=None, help="Min importance for network figure")
    parser.add_argument("--top-tfs", type=int, default=20, help="Top TFs to show in bar plots")
    parser.add_argument("--heatmap-top-tfs", type=int, default=25, help="Top TFs to show in heatmap")
    parser.add_argument("--min-label-degree", type=int, default=2, help="Min degree to label nodes")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output")
    args = parser.parse_args()

    set_plot_style()
    grn_path = resolve_grn_path(args.grn)
    logger.info("Loading GRN from %s", grn_path)
    grn = pd.read_csv(grn_path)

    required = {"TF", "target", "importance"}
    if not required.issubset(grn.columns):
        missing = required - set(grn.columns)
        raise ValueError(f"GRN file missing columns: {sorted(missing)}")

    output_dir = Path(args.output)
    build_network_figure(
        grn,
        output_dir=output_dir,
        top_edges=args.top_edges,
        min_importance=args.min_importance,
        min_label_degree=args.min_label_degree,
        dpi=args.dpi,
    )
    build_tf_activity_heatmap(grn, output_dir=output_dir, top_tfs=args.heatmap_top_tfs, dpi=args.dpi)
    build_tf_class_dotplot(grn, output_dir=output_dir, top_tfs=args.heatmap_top_tfs, dpi=args.dpi)
    build_top_regulators_bars(grn, output_dir=output_dir, top_tfs=args.top_tfs, dpi=args.dpi)
    build_importance_distribution(grn, output_dir=output_dir, dpi=args.dpi)
    write_legend_file(output_dir, grn_path)
    logger.info("Figures saved to %s", output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
