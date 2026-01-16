import logging
from typing import Dict

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_graph(network: pd.DataFrame, regulator_col: str = "regulator", target_col: str = "target") -> nx.DiGraph:
    """
    Build a directed graph from an edge list.

    Args:
        network: GRN edges DataFrame
        regulator_col: Column name for regulators
        target_col: Column name for targets

    Returns:
        Directed graph
    """
    graph = nx.from_pandas_edgelist(network, source=regulator_col, target=target_col, create_using=nx.DiGraph)
    logger.info("Graph constructed with %s nodes and %s edges", graph.number_of_nodes(), graph.number_of_edges())
    return graph


def centrality_metrics(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Compute basic centrality metrics for a GRN.

    Args:
        graph: Directed graph

    Returns:
        Dictionary of metric name to node score mapping
    """
    metrics = {
        "in_degree": dict(graph.in_degree()),
        "out_degree": dict(graph.out_degree()),
        "pagerank": nx.pagerank(graph),
    }
    return metrics
