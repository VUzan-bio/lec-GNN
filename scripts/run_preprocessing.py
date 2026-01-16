import logging
from pathlib import Path
from typing import Dict

import yaml

from src.data.preprocessing import LECDataPreprocessor

logger = logging.getLogger(__name__)


def load_sources(config_path: Path) -> Dict:
    """Load dataset sources from YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    """Run preprocessing for all datasets in config."""
    logging.basicConfig(level=logging.INFO)
    config = load_sources(Path("config/data_sources.yaml"))

    preprocessor = LECDataPreprocessor()

    for dataset in config.get("arrayexpress", []):
        accession = dataset["accession"]
        species = dataset.get("species", "mouse")
        data_dir = Path("data/raw") / accession
        if not data_dir.exists():
            logger.warning("Missing raw directory for %s", accession)
            continue
        preprocessor.process_dataset(data_dir, accession, species, source="arrayexpress")

    for dataset in config.get("geo", []):
        accession = dataset["accession"]
        species = dataset.get("species", "human")
        data_dir = Path("data/raw") / accession
        if not data_dir.exists():
            logger.warning("Missing raw directory for %s", accession)
            continue
        preprocessor.process_dataset(data_dir, accession, species, source="geo")


if __name__ == "__main__":
    main()
