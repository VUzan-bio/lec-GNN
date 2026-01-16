# LEC Trafficking GNN Project

Project for modeling lymphatic endothelial cell (LEC) trafficking competence using scRNA-seq, gene regulatory network (GRN) inference, and graph neural networks (GNNs).

Scientific goal: infer GRNs linking transcription factors, glycosylation enzymes, and trafficking genes, then predict LEC phenotypes that support dendritic cell trafficking.

## Datasets
- E-MTAB-8414 (ArrayExpress/BioStudies): mouse lymph node LEC subsets
- GSE148730 (GEO): human dermal LEC activation states
- GSE282417 (GEO): human LEC shear stress response

## Quickstart
1) Create environment

```bash
conda env create -f environment.yml
conda activate lec-trafficking-gnn
```

2) Download data

```bash
python -m src.data.download --config config/data_sources.yaml --output data/raw
```

3) Preprocess data

```bash
python -m src.data.preprocessing --data_dir data/raw/E-MTAB-8414 --accession E-MTAB-8414 --species mouse --source arrayexpress
```

## Repository layout
- `config/`: dataset, preprocessing, and model parameters
- `data/`: raw, processed, external databases, metadata
- `src/`: modular code for data, GRN, models, visualization, utils
- `notebooks/`: exploratory and reproducible analysis
- `scripts/`: command line pipelines
- `results/`: figures, tables, models, logs
- `tests/`: unit tests

## Notes
- Do not commit large data files under `data/` or `results/`.
- Parameters are config-driven; edit `config/*.yaml` to customize.

## Testing

```bash
pytest
mypy src
```

## License
MIT
