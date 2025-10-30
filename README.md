# NVIDIA Stock Prediction Model

Machine learning workflow for forecasting NVIDIA (NVDA) stock prices with reproducible data ingestion, feature engineering, model training, evaluation, and interactive reporting.

## Highlights
- Modular Python package (`src/`) covering data loaders, feature builders, models, pipelines, and evaluation utilities.
- Reproducible experiment layout that separates raw/interim/processed data, trained artefacts, and reports.
- Configuration-driven data ingestion using YAML files for tickers, time ranges, and external indicators.
- Plotly Dash dashboard scaffold (`reports/dashboard/`) for visualising predictions, residuals, and rolling metrics.
- Testing scaffold (`tests/`) to keep pipelines stable as the project grows.

## Repository Structure
```
nvidia_stock_predictor/
├── configs/                # YAML configuration files (data sources, pipeline params, experiment settings)
├── data/
│   ├── external/           # External signals (macro indicators, sentiment, etc.) ← populated by scripts
│   ├── interim/            # Intermediate artefacts produced during preprocessing
│   ├── processed/          # Model-ready datasets (train/test splits, feature matrices)
│   └── raw/                # Raw downloads (NVDA prices, benchmarks) ← populated by scripts
├── docs/                   # Project documentation, experiment notes, design decisions
├── experiments/            # Experiment tracking exports (MLflow, W&B, custom logs)
├── logs/                   # Runtime logs from data pipelines, training, and evaluation
├── notebooks/              # Exploratory analysis and prototyping notebooks
├── reports/
│   ├── dashboard/          # Interactive dashboard app (Plotly Dash)
│   └── figures/            # Static charts and generated report assets
├── scripts/                # CLI scripts (data download, training orchestration, batch inference)
├── src/
│   ├── config/             # Config loaders, schema validation, environment helpers
│   ├── data/               # Data access layer (downloaders, dataset builders)
│   ├── evaluation/         # Metrics, backtesting, visualisation utilities
│   ├── features/           # Feature engineering transforms
│   ├── models/             # Model architectures, training loops, hyperparameter search
│   ├── pipelines/          # End-to-end pipelines tying data → features → model → evaluation
│   └── utils/              # Shared utilities (logging setup, time handling, etc.)
├── tests/                  # Unit and integration tests
├── trained_models/         # Persisted model weights and metadata
└── requirements.txt        # Python dependencies pinned for reproducibility
```

## Getting Started

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Load secrets (FRED API key, etc.)
source .env
```

### Configure Data Sources
1. Copy `configs/data_sources.example.yaml` to `configs/data_sources.yaml`.
2. Adjust ticker symbols, date ranges, and external series to match your experiment.
3. If you plan to pull macro indicators (e.g., from FRED), supply API keys via environment variables or a `.env` file referenced in the config.

### Download Data
```bash
python scripts/download_data.py --config configs/data_sources.yaml
```
This command:
1. Loads the YAML configuration.
2. Downloads NVDA OHLCV data via Yahoo Finance (requires internet access).
3. Optionally fetches additional benchmarks and macro indicators.
4. Stores raw CSV files under `data/raw/` and external signals under `data/external/`.

### Next Steps
- Implement feature engineering transforms in `src/features/`.
- Create training pipeline(s) under `src/pipelines/` and persist models to `trained_models/`.
- Add evaluation routines in `src/evaluation/` to compute metrics (RMSE, MAPE, directional accuracy).
- Update `reports/dashboard/` with Dash layouts consuming evaluation artefacts.
- Write tests under `tests/` and run with `pytest`.

## Testing
```bash
pytest
```
Add new tests alongside modules as they are implemented to prevent regressions.

## Contributing & Roadmap
- Document decisions in `docs/adr/` as architecture evolves.
- Track experiments via `experiments/` exports or integrate MLflow/W&B.
- Expand configuration schemas to cover hyperparameter sweeps and backtests.

Feel free to open issues or submit pull requests once the repository is published.
