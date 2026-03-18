# AutoTS Load Forecasting Pipeline

Automated energy load forecasting using [AutoTS](https://github.com/winedarksea/AutoTS) with rolling window evaluation and template-based model deployment.

## Overview

This pipeline forecasts building/residential energy consumption (grid import in kW) using AutoTS's genetic algorithm-based model selection. It trains on January data and evaluates rolling forecasts across February–December, retraining weekly with the best models persisted via AutoTS's template system.

### Key Features

- **Rolling window forecasting** with configurable horizon and stride (default: 6h)
- **Weekly retraining** using AutoTS's genetic algorithm search across dozens of model classes
- **Template-based deployment**: best models are exported/imported between runs (`import_template` / `export_template`)
- **Probabilistic forecasts**: upper/lower bounds via `prediction_interval` (default: 95%)
- **6 evaluation metrics**: RMSE, NRMSE, MAPE, Pearson r, prediction interval coverage, interval width
- **SVG plots**: error difference plot + actual vs. forecast comparison with confidence band

## Quick Start

```bash
pip install -r requirements.txt
python autots_forecasting_pipeline.py
```

## Configuration

Edit the constants at the top of `autots_forecasting_pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `CSV_NAME` | `"residential5.csv"` | Input CSV file in `data/` |
| `FORECAST_YEAR` | `2016` | Year to forecast |
| `HORIZON_HOURS` | `6` | Forecast horizon in hours |
| `FORECAST_EVERY_HOURS` | `6` | Stride — how often a new forecast is issued |
| `MODEL_LIST` | `"default"` | AutoTS model list (`"default"`, `"fast"`, `"superfast"`, etc.) |
| `ENSEMBLE` | `"simple"` | Ensemble method (`"simple"` = BestN average) |
| `MAX_GENERATIONS` | `5` | Genetic algorithm generations |
| `NUM_VALIDATIONS` | `2` | Cross-validation splits (backwards method) |
| `PREDICTION_INTERVAL` | `0.95` | Confidence level for prediction bounds |
| `N_JOBS` | `-1` | Parallel jobs (-1 = all cores) |

Also update `TIMESTAMP_COL` and `TARGET_COL` to match your CSV column names.

## Project Structure

```
├── autots_forecasting_pipeline.py   # Main forecasting pipeline
├── autots_example.py                # Standalone AutoTS demo with synthetic data
├── requirements.txt
├── data/                            # Input CSV files (semicolon-delimited, 15-min resolution)
│   ├── residential1.csv
│   ├── building_0.csv
│   └── ...
└── results/                         # Auto-generated per dataset
    └── <dataset_name>/
        ├── <dataset>_best_models.json          # AutoTS template (best models)
        ├── <dataset>_results_H6h_2016_*.json   # Metrics JSON
        ├── <dataset>_timeseries_H6h_2016_*.csv # Actual + predicted + bounds
        ├── <dataset>_autots_comparison_*.svg    # Comparison plot
        └── <dataset>_autots_difference_*.svg    # Error plot
```

## Input Data Format

Semicolon-delimited CSV with at least two columns:

| Column | Example |
|---|---|
| `utc_timestamp` | `2016-01-01 00:00:00+01:00` |
| Target column (e.g. `DE_KN_residential5_grid_import`) | `0.148` |

## Dependencies

See [requirements.txt](requirements.txt). Core dependency is [AutoTS](https://github.com/winedarksea/AutoTS) (MIT license).
