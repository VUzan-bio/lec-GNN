import logging
from typing import Dict, List

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class GRNAnalyzer:
    """
    Network topology analysis and hub gene identification.
    """

    def __init__(self, grn: pd.DataFrame) -> None:
        """
        Args:
            grn: GRN DataFrame with [TF, target, importance]
        """
        self.grn = grn
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """Convert GRN to a directed graph."""
        graph = nx.DiGraph()
        for _, row in self.grn.iterrows():
            attrs = {"weight": row["importance"]}
            if "edge_type" in self.grn.columns:
                attrs["edge_type"] = row["edge_type"]
            graph.add_edge(row["TF"], row["target"], **attrs)

        logger.info("Built graph: %s nodes, %s edges", graph.number_of_nodes(), graph.number_of_edges())
        return graph

    def compute_centrality_metrics(self, undirected_betweenness: bool = False) -> pd.DataFrame:
        """
        Compute centrality measures for each gene.

        Returns:
            DataFrame with centrality scores per gene
        """
        logger.info("Computing network centrality metrics")

        pagerank = nx.pagerank(self.graph, weight="weight")
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        betweenness_graph = self.graph.to_undirected() if undirected_betweenness else self.graph
        betweenness_graph = betweenness_graph.copy()
        for _, _, data in betweenness_graph.edges(data=True):
            weight = data.get("weight", 1.0)
            data["distance"] = 1.0 / weight if weight and weight > 0 else 1.0
        betweenness = nx.betweenness_centrality(betweenness_graph, weight="distance")

        genes = list(self.graph.nodes())
        centrality_df = pd.DataFrame(
            {
                "gene": genes,
                "pagerank": [pagerank[gene] for gene in genes],
                "in_degree": [in_degree[gene] for gene in genes],
                "out_degree": [out_degree[gene] for gene in genes],
                "betweenness": [betweenness[gene] for gene in genes],
            }
        )

        glyco_genes = [
            "Galnt1",
            "Galnt2",
            "St3gal1",
            "Gcnt1",
            "C1galt1",
            "GALNT1",
            "GALNT2",
            "ST6GALNAC3",
            "A3GALT2",
            "B3GNT7",
            "GCNT1",
            "C1GALT1",
        ]
        trafficking_genes = [
            "Ccl21a",
            "Lyve1",
            "Cd274",
            "Alcam",
            "VCAM1",
            "PLPP3",
            "F3",
            "CCL21",
            "CCL21A",
            "LYVE1",
            "CD274",
            "ALCAM",
        ]

        centrality_df["gene_type"] = centrality_df["gene"].apply(
            lambda gene: "glyco" if gene in glyco_genes else ("trafficking" if gene in trafficking_genes else "other")
        )

        centrality_df = centrality_df.sort_values("pagerank", ascending=False)
        if not centrality_df.empty:
            top_gene = centrality_df.iloc[0]
            logger.info("  Top hub gene: %s (PageRank=%.4f)", top_gene["gene"], top_gene["pagerank"])

        return centrality_df

    def identify_hubs(self, centrality_df: pd.DataFrame, percentile: float = 0.90) -> pd.DataFrame:
        """
        Identify hub genes (top X% by PageRank).

        Args:
            centrality_df: Output from compute_centrality_metrics()
            percentile: Threshold for hub definition (0.90 = top 10%)

        Returns:
            DataFrame of hub genes
        """
        threshold = centrality_df["pagerank"].quantile(percentile)
        hubs = centrality_df[centrality_df["pagerank"] >= threshold].copy()

        logger.info("Identified %s hub genes (top %.0f%%)", len(hubs), 100 * (1 - percentile))

        n_glyco = (hubs["gene_type"] == "glyco").sum()
        n_traffic = (hubs["gene_type"] == "trafficking").sum()

        logger.info("  Glycosylation hubs: %s", n_glyco)
        logger.info("  Trafficking hubs: %s", n_traffic)

        return hubs

    def detect_communities(self) -> Dict[str, int]:
        """
        Detect regulatory modules using Louvain algorithm.

        Returns:
            Dictionary: {gene: community_id}
        """
        try:
            import community.community_louvain as community_louvain
        except ImportError as exc:
            raise ImportError("python-louvain is required for community detection.") from exc

        logger.info("Detecting regulatory communities")

        undirected = self.graph.to_undirected()
        communities = community_louvain.best_partition(undirected, weight="weight")

        n_communities = len(set(communities.values()))
        logger.info("  Found %s communities", n_communities)

        trafficking_genes = ["Ccl21a", "Lyve1", "Cd274"]
        trafficking_communities = [communities.get(gene) for gene in trafficking_genes if gene in communities]

        logger.info("  Trafficking genes in communities: %s", sorted(set(trafficking_communities)))

        return communities


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze GRN topology")
    parser.add_argument("--grn", required=True, help="Path to GRN CSV file")
    parser.add_argument("--output", required=True, help="Output path prefix for analysis results")
    parser.add_argument(
        "--undirected_betweenness",
        action="store_true",
        help="Compute betweenness on an undirected version of the network",
    )

    args = parser.parse_args()

    grn_df = pd.read_csv(args.grn)

    analyzer = GRNAnalyzer(grn_df)
    centrality = analyzer.compute_centrality_metrics(undirected_betweenness=args.undirected_betweenness)
    hubs = analyzer.identify_hubs(centrality, percentile=0.90)
    _ = analyzer.detect_communities()

    output_prefix = args.output.replace(".csv", "")
    centrality.to_csv(f"{output_prefix}_centrality.csv", index=False)
    hubs.to_csv(f"{output_prefix}_hubs.csv", index=False)

    print("\nNetwork analysis complete")
    print(f"  Hub genes saved to {output_prefix}_hubs.csv")
