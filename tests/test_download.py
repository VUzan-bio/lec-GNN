from pathlib import Path

from src.data.download import DatasetDownloader


def test_config_load(tmp_path: Path) -> None:
    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text("arrayexpress: []\ngeo: []\n", encoding="utf-8")

    downloader = DatasetDownloader(config_path=str(config_path), output_dir=str(tmp_path / "raw"))

    assert downloader.config["arrayexpress"] == []
    assert downloader.config["geo"] == []
