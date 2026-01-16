import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def run_grnboost2(
    expression: pd.DataFrame,
    tf_list: List[str],
    random_state: int = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Run GRNBoost2 to infer gene regulatory networks.

    Args:
        expression: Gene expression matrix (cells x genes)
        tf_list: List of transcription factors to use as regulators
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs

    Returns:
        DataFrame of inferred regulatory edges
    """
    try:
        from arboreto.algo import grnboost2
    except ImportError as exc:
        raise ImportError("arboreto is required for GRNBoost2. Install arboreto.") from exc

    logger.info("Running GRNBoost2 on matrix with shape %s", expression.shape)
    network = grnboost2(expression_data=expression, tf_names=tf_list, seed=random_state, n_jobs=n_jobs)
    return network
