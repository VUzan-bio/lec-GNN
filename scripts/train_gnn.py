import argparse
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Placeholder entry point for GNN training."""
    parser = argparse.ArgumentParser(description="Train GAT model for LEC trafficking prediction")
    parser.add_argument("--config", default="config/model_params.yaml", help="Path to model config")
    parser.add_argument("--data", default="data/processed", help="Path to processed data")

    _ = parser.parse_args()

    raise NotImplementedError("Implement training in scripts/train_gnn.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
