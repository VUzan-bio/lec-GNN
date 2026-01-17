import argparse
import logging
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def annotate_gene_type(gene: str, tf_set: set, glyco_set: set, traffic_set: set) -> str:
    """Assign gene to TF/glyco/trafficking/other."""
    if gene in tf_set:
        return "TF"
    if gene in glyco_set:
        return "glyco"
    if gene in traffic_set:
        return "trafficking"
    return "other"


def add_distance_weights(graph: nx.Graph) -> None:
    """Convert importance weights to distance for shortest-path metrics."""
    for _, _, data in graph.edges(data=True):
        weight = data.get("importance", 1.0)
        data["distance"] = 1.0 / weight if weight and weight > 0 else 1.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Full centrality analysis for combined GRN.")
    parser.add_argument(
        "--grn",
        default="results/grn/GSE282417_grn_with_coexpr.csv",
        help="Path to combined GRN CSV",
    )
    parser.add_argument(
        "--output",
        default="results/grn/GSE282417_centrality_combined.csv",
        help="Output CSV for centrality table",
    )
    parser.add_argument(
        "--community_output",
        default="results/grn/GSE282417_communities.csv",
        help="Output CSV for community membership",
    )
    parser.add_argument(
        "--cascade_output",
        default="results/grn/GSE282417_glyco_cascades.csv",
        help="Output CSV for glyco cascades",
    )
    args = parser.parse_args()

    grn_path = Path(args.grn)
    grn = pd.read_csv(grn_path)
    if "TF" not in grn.columns and "regulator" in grn.columns:
        grn = grn.rename(columns={"regulator": "TF"})

    graph = nx.from_pandas_edgelist(
        grn,
        source="TF",
        target="target",
        edge_attr="importance",
        create_using=nx.DiGraph(),
    )

    logger.info("Network: %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges())

    pagerank = nx.pagerank(graph, weight="importance", alpha=0.85)

    undirected = graph.to_undirected()
    add_distance_weights(undirected)
    betweenness = nx.betweenness_centrality(undirected, weight="distance", normalized=True)
    closeness = nx.closeness_centrality(undirected, distance="distance")

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    centrality_df = pd.DataFrame(
        {
            "gene": list(pagerank.keys()),
            "pagerank": list(pagerank.values()),
            "betweenness": [betweenness[gene] for gene in pagerank.keys()],
            "closeness": [closeness[gene] for gene in pagerank.keys()],
            "in_degree": [in_degree[gene] for gene in pagerank.keys()],
            "out_degree": [out_degree[gene] for gene in pagerank.keys()],
        }
    )
    centrality_df["total_degree"] = centrality_df["in_degree"] + centrality_df["out_degree"]

    tf_list = ["NFKB1", "PROX1", "SOX18", "FOXC2", "GATA2"]
    glyco_genes = ["ST6GALNAC3", "A3GALT2", "GALNT1", "GALNT2", "B3GNT7", "GCNT1"]
    trafficking_genes = ["VCAM1", "PLPP3", "F3", "CCL21A", "LYVE1", "CD53", "TNFRSF1B"]

    tf_set = set(tf_list)
    glyco_set = set(glyco_genes)
    traffic_set = set(trafficking_genes)

    centrality_df["gene_type"] = centrality_df["gene"].apply(
        lambda gene: annotate_gene_type(gene, tf_set, glyco_set, traffic_set)
    )

    max_degree = float(centrality_df["total_degree"].max()) or 1.0
    centrality_df["hub_score"] = (
        centrality_df["betweenness"] * 0.4
        + centrality_df["pagerank"] * 100 * 0.3
        + (centrality_df["total_degree"] / max_degree) * 0.3
    )

    centrality_df = centrality_df.sort_values("hub_score", ascending=False)

    print("\n" + "=" * 80)
    print("TOP 20 HUB GENES (ranked by composite hub score)")
    print("=" * 80)
    print(
        centrality_df[
            ["gene", "hub_score", "betweenness", "pagerank", "in_degree", "out_degree", "gene_type"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("GLYCOSYLATION GENES (ranked by hub score)")
    print("=" * 80)
    glyco_df = centrality_df[centrality_df["gene_type"] == "glyco"].copy()
    if not glyco_df.empty:
        print(glyco_df[["gene", "hub_score", "betweenness", "in_degree", "out_degree"]].to_string(index=False))
    else:
        print("No glyco genes found in network")

    print("\n" + "=" * 80)
    print("TRAFFICKING GENES (ranked by hub score)")
    print("=" * 80)
    traffic_df = centrality_df[centrality_df["gene_type"] == "trafficking"].copy()
    if not traffic_df.empty:
        print(
            traffic_df[["gene", "hub_score", "betweenness", "in_degree", "out_degree"]].to_string(index=False)
        )
    else:
        print("No trafficking genes found in network")

    print("\n" + "=" * 80)
    print("BRIDGE GENES (high betweenness + balanced in/out)")
    print("=" * 80)
    bridge_df = centrality_df[
        (centrality_df["betweenness"] > centrality_df["betweenness"].quantile(0.75))
        & (centrality_df["in_degree"] >= 1)
        & (centrality_df["out_degree"] >= 1)
    ].copy()
    if not bridge_df.empty:
        print(bridge_df[["gene", "betweenness", "in_degree", "out_degree", "gene_type"]].to_string(index=False))
    else:
        print("No bridge genes found")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    centrality_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n" + "=" * 80)
    print("CO-EXPRESSION MODULES (Louvain communities)")
    print("=" * 80)
    communities = []
    try:
        communities = list(nx.community.louvain_communities(undirected, weight="importance", seed=42))
    except Exception as exc:
        logger.warning("Louvain community detection skipped: %s", exc)

    if communities:
        rows: List[Dict[str, object]] = []
        for i, community in enumerate(communities, 1):
            genes_in_module = sorted(list(community))
            gene_types = [annotate_gene_type(gene, tf_set, glyco_set, traffic_set) for gene in genes_in_module]
            print(f"\nModule {i} ({len(genes_in_module)} genes):")
            print(f"  Genes: {', '.join(genes_in_module)}")
            print(
                "  Types: TF=%s, glyco=%s, trafficking=%s, other=%s",
                gene_types.count("TF"),
                gene_types.count("glyco"),
                gene_types.count("trafficking"),
                gene_types.count("other"),
            )
            if gene_types.count("glyco") >= 2:
                print("  GLYCO-ENRICHED MODULE")
            if gene_types.count("trafficking") >= 2:
                print("  TRAFFICKING-ENRICHED MODULE")
            for gene in genes_in_module:
                rows.append({"gene": gene, "community": i})

        community_path = Path(args.community_output)
        community_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(community_path, index=False)
        print(f"\nSaved communities to {community_path}")

    print("\n" + "=" * 80)
    print("TF -> GLYCO -> TARGET CASCADES (2-hop paths)")
    print("=" * 80)
    paths_2hop = []
    for tf in tf_list:
        for glyco in glyco_genes:
            tf_to_glyco = grn[(grn["TF"] == tf) & (grn["target"] == glyco)]
            if not tf_to_glyco.empty:
                glyco_targets = grn[grn["TF"] == glyco]
                for _, edge2 in glyco_targets.iterrows():
                    target = edge2["target"]
                    paths_2hop.append(
                        {
                            "path": f"{tf} -> {glyco} -> {target}",
                            "TF": tf,
                            "glyco": glyco,
                            "target": target,
                            "target_type": annotate_gene_type(target, tf_set, glyco_set, traffic_set),
                            "edge1_importance": tf_to_glyco.iloc[0]["importance"],
                            "edge2_importance": edge2["importance"],
                            "path_strength": tf_to_glyco.iloc[0]["importance"] * edge2["importance"],
                        }
                    )

    if paths_2hop:
        paths_df = pd.DataFrame(paths_2hop).sort_values("path_strength", ascending=False)
        print(f"Found {len(paths_df)} TF -> glyco -> target cascades")
        print("\nTop 10 cascades:")
        print(paths_df[["path", "path_strength", "target_type"]].head(10).to_string(index=False))

        traffic_cascades = paths_df[paths_df["target_type"] == "trafficking"]
        if not traffic_cascades.empty:
            print(f"\nFound {len(traffic_cascades)} TF -> glyco -> TRAFFICKING cascades:")
            print(traffic_cascades[["path", "path_strength"]].to_string(index=False))

        cascade_path = Path(args.cascade_output)
        cascade_path.parent.mkdir(parents=True, exist_ok=True)
        paths_df.to_csv(cascade_path, index=False)
        print(f"\nSaved cascades to {cascade_path}")
    else:
        print("No TF -> glyco -> target cascades found")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Expand to 500-1000 genes to capture glyco -> trafficking edges")
    print("2. Add pathway genes (GO:0006486 glycosylation, GO:0050904 diapedesis)")
    print("3. Build GNN using this network structure")


if __name__ == "__main__":
    main()
