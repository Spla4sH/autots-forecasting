"""AutoTS Rolling Window Forecasting Pipeline for energy load data.

Uses the AutoTS library (https://github.com/winedarksea/AutoTS) to perform
automated time series forecasting with genetic algorithm-based model selection,
template-based deployment, and probabilistic (upper/lower bound) forecasts.
"""

import warnings
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime as dt

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import pearsonr
from autots import AutoTS

# ============================ CONFIGURATION ============================
CSV_NAME = "residential5.csv"
FORECAST_YEAR = 2016

HORIZON_HOURS = 6  # Forecast horizon in hours
FORECAST_EVERY_HOURS = 6  # Stride: how often a new forecast is issued

N_JOBS = -1  # Parallel jobs for AutoTS (-1 = all cores)

# AutoTS model search configuration
# See: https://github.com/winedarksea/AutoTS#basic-use
MODEL_LIST = "default"  # "default" includes all robust model classes
ENSEMBLE = "simple"  # "simple" = BestN average of top models (reliable baseline)
MAX_GENERATIONS = 5  # Genetic algorithm generations for model search
NUM_VALIDATIONS = 2  # Cross-validation splits beyond initial eval ("backwards" method)
PREDICTION_INTERVAL = 0.95  # Confidence level for upper/lower forecast bounds

# ============================ SETUP & LOGGING ============================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("autots-pipeline")

# ============================ PATHS ============================
DATASET_NAME = Path(CSV_NAME).stem

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR
except NameError:
    PROJECT_ROOT = Path(".").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_BASE_DIR = PROJECT_ROOT / "results"

# CSV column names (must match the input CSV)
TIMESTAMP_COL = "utc_timestamp"
TARGET_COL = "DE_KN_residential5_grid_import"

RESULTS_DIR = RESULTS_BASE_DIR / DATASET_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / CSV_NAME
# AutoTS template file for persisting best models across runs
# See: https://winedarksea.github.io/AutoTS/build/html/source/tutorial.html#deployment-and-template-import-export
TEMPLATE_PATH = RESULTS_DIR / f"{DATASET_NAME}_best_models.json"


# ============================ DATA STRUCTURES ============================
@dataclass
class Metrics:
    """Forecast evaluation metrics."""

    rmse: float
    nrmse: float
    mape: float
    pearson: float
    coverage: float  # % of actuals within the prediction interval
    interval_width_mean: float  # Average width of the prediction interval


# ============================ DATA LOADING ============================
def load_and_prepare_data(csv_path: Path, time_col: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, deduplicate, interpolate missing values.

    AutoTS expects a wide-format DataFrame with a tz-naive DatetimeIndex.
    The frequency is inferred automatically via pd.infer_freq.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    log.info(f"Loading data from: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", usecols=[time_col, TARGET_COL])
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    # AutoTS requires timezone-naive datetimes
    df.index = df.index.tz_localize(None)

    # Fill gaps via linear interpolation + boundary fill
    df = df.interpolate(method="linear").bfill().ffill()

    freq = pd.infer_freq(df.index)
    if freq:
        log.info(f"Detected frequency: {freq}")
    else:
        log.warning("Could not infer frequency automatically")

    # Derive steps per hour (e.g. 4 for 15-min data) for forecast_length calculation
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    log.info(f"Data loaded: {len(df)} points, {df.index[0]} to {df.index[-1]}")
    log.info(
        f"Resolution: {minutes_per_step:.1f} min/step ({steps_per_hour} steps/hour)"
    )

    return df


# ============================ METRICS ============================
def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    upper_forecast: pd.Series = None,
    lower_forecast: pd.Series = None,
) -> Metrics:
    """Compute forecast evaluation metrics.

    Calculates RMSE, NRMSE (%-normalized by mean), MAPE, Pearson correlation,
    and — if prediction intervals are provided — coverage and mean interval width.
    """
    actual_vals = actual.values.flatten()
    pred_vals = predicted.values.flatten()

    min_len = min(len(actual_vals), len(pred_vals))
    if min_len == 0:
        raise ValueError("No overlapping data between actual and predicted")

    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]

    rmse_val = np.sqrt(np.mean((actual_vals - pred_vals) ** 2))

    # NRMSE: RMSE normalized by the mean of actuals (in %)
    mean_val = float(np.mean(actual_vals))
    nrmse_val = (rmse_val / mean_val * 100) if abs(mean_val) > 1e-6 else float("inf")

    # MAPE with epsilon guard against division by zero
    eps = 1e-6
    mape_val = float(
        np.mean(
            np.abs((actual_vals - pred_vals) / np.maximum(np.abs(actual_vals), eps))
        )
        * 100
    )

    pearson_val = (
        float(pearsonr(actual_vals, pred_vals)[0])
        if len(actual_vals) > 1
        else float("nan")
    )

    # Prediction interval metrics (if bounds are provided)
    coverage_val = 0.0
    interval_width_mean = 0.0

    if upper_forecast is not None and lower_forecast is not None:
        upper_vals = upper_forecast.values.flatten()[:min_len]
        lower_vals = lower_forecast.values.flatten()[:min_len]

        # Coverage: % of actuals falling within [lower, upper]
        within_interval = (actual_vals >= lower_vals) & (actual_vals <= upper_vals)
        coverage_val = float(np.mean(within_interval) * 100)

        interval_width_mean = float(np.mean(upper_vals - lower_vals))

    return Metrics(
        rmse=float(rmse_val),
        nrmse=float(nrmse_val),
        mape=float(mape_val),
        pearson=float(pearson_val),
        coverage=coverage_val,
        interval_width_mean=interval_width_mean,
    )


# ============================ PLOTTING ============================
def plot_simple_difference(
    val_series: pd.Series,
    prediction: pd.Series,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int,
) -> None:
    """Plot forecast error (predicted - actual) over time."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    # Align indices
    common_idx = val_series.index.intersection(prediction.index)
    val_aligned = val_series.loc[common_idx]
    pred_aligned = prediction.loc[common_idx]

    diff = pred_aligned.values - val_aligned.values

    ax.plot(
        common_idx,
        diff.flatten(),
        color="darkblue",
        linewidth=0.5,
        alpha=0.7,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0)

    ax.set_title(
        f"Vorhersagefehler {forecast_year} (Stündlich) - RMSE: {metrics.rmse:.2f} kW | "
        f"NRMSE: {metrics.nrmse:.1f}% | MAPE: {metrics.mape:.1f}% | Pearson: {metrics.pearson:.3f}",
        fontsize=14,
    )
    ax.set_ylabel("Fehler (kW) [Prognose - Ist]", fontsize=14)
    ax.set_xlabel("Datum", fontsize=14)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    log.info(f"Difference plot saved: {save_path}")
    plt.close(fig)


def plot_comparison(
    val_series: pd.Series,
    prediction: pd.Series,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int,
    upper_forecast: pd.Series = None,
    lower_forecast: pd.Series = None,
) -> None:
    """Plot actual vs. forecast with optional prediction interval band."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # Align indices
    common_idx = val_series.index.intersection(prediction.index)
    val_aligned = val_series.loc[common_idx]
    pred_aligned = prediction.loc[common_idx]

    # Shaded prediction interval band (if bounds available)
    if upper_forecast is not None and lower_forecast is not None:
        upper_aligned = upper_forecast.loc[common_idx]
        lower_aligned = lower_forecast.loc[common_idx]

        ax.fill_between(
            common_idx,
            lower_aligned.values.flatten(),
            upper_aligned.values.flatten(),
            color="#ff7f0e",
            alpha=0.2,
            label="95% Konfidenzintervall",
            zorder=1,
        )

    ax.plot(
        val_aligned.index,
        val_aligned.values,
        color="#1f4e79",
        linewidth=1.0,
        alpha=0.95,
        label="Tatsächliche Last",
        zorder=3,
    )
    ax.plot(
        pred_aligned.index,
        pred_aligned.values,
        color="#d62728",
        linewidth=1.0,
        alpha=0.95,
        label="AutoTS Prognose",
        zorder=2,
    )

    title_text = (
        f"AutoTS Jahresprognose ({CSV_NAME}) {forecast_year} - Horizont: {HORIZON_HOURS}h - "
        f"RMSE: {metrics.rmse:.2f} kW | NRMSE: {metrics.nrmse:.1f}% | "
        f"MAPE: {metrics.mape:.1f}% | Pearson: {metrics.pearson:.3f}"
    )
    if metrics.coverage > 0:
        title_text += f" | Coverage: {metrics.coverage:.1f}%"
    ax.set_title(title_text, fontsize=18, fontweight="bold", pad=25)
    ax.set_ylabel("Last (kW)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Monate", fontsize=14, fontweight="bold")

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=14,
        framealpha=0.9,
    )
    legend.get_frame().set_facecolor("white")

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=2))

    ax.set_xlim(val_aligned.index[0], val_aligned.index[-1])
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0.02)

    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5, color="#bdc3c7")
    ax.grid(
        True, alpha=0.2, linestyle="-", linewidth=0.3, color="#bdc3c7", which="minor"
    )

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#34495e")
    ax.spines["bottom"].set_color("#34495e")
    ax.tick_params(colors="#34495e", which="both")

    plt.tight_layout()
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    log.info(f"Comparison plot saved: {save_path}")
    plt.close(fig)


# ============================ ROLLING WINDOW FORECASTING ============================
def rolling_window_forecast(
    df: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    horizon_hours: int,
    forecast_every_hours: int,
    steps_per_hour: int = 4,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling window forecasting with periodic retraining.

    Slides a fixed-length training window across the dataset. The model is
    retrained weekly using AutoTS's genetic algorithm search. Between retraining
    cycles, the existing model is reused for predictions via model.predict().

    Template system (see AutoTS deployment docs):
    - import_template(method="only"): restrict search to previously saved best models
    - export_template(models="best", n=15): persist top 15 models for future runs
    - predict() returns a PredictionObject with .forecast, .upper_forecast, .lower_forecast

    Args:
        df: Full dataset (used to slice the rolling training window).
        train_data: Initial training data (January) that defines window length.
        test_data: Test period (Feb-Dec) to forecast over.
        horizon_hours: Forecast horizon in hours.
        forecast_every_hours: Stride — how often a new forecast is issued.
        steps_per_hour: Data points per hour (e.g. 4 for 15-min resolution).

    Returns:
        Tuple of (point_forecasts, upper_bounds, lower_bounds) as pd.Series.
    """
    log.info("=== Starting Rolling Window Forecasting ===")

    # Weekly retraining cycle (e.g. every ~28 iterations at 6h stride)
    retrain_every = 7 * 24 // forecast_every_hours

    # Fixed-length sliding window: always use the most recent N points
    train_window_length = len(train_data)
    test_start = test_data.index[0]
    test_end = test_data.index[-1]

    current_forecast_start = test_start
    current_train_end = train_data.index[-1]

    # Convert hours to data points (e.g. 6h * 4 steps/h = 24 steps)
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    forecast_length_steps = horizon_hours * steps_per_hour
    forecast_stride_steps = forecast_every_hours * steps_per_hour

    log.info(f"Forecast length: {horizon_hours}h = {forecast_length_steps} steps")
    log.info(f"Forecast stride: {forecast_every_hours}h = {forecast_stride_steps} steps")
    log.info(
        f"Fixed rolling window: {train_window_length} points "
        f"({train_window_length/(steps_per_hour*24):.1f} days)"
    )
    log.info(f"Retraining every {retrain_every} iterations (~1 week)")

    # Check for existing template (saved best models from prior runs)
    use_template = TEMPLATE_PATH.exists()

    if use_template:
        log.info(f"Template found: {TEMPLATE_PATH.name}")
    else:
        log.info("No template — running full model search")

    # Expected iterations for progress reporting
    test_duration_minutes = (test_end - test_start).total_seconds() / 60
    expected_iterations = int(test_duration_minutes / (forecast_every_hours * 60)) + 1
    log.info(f"Expected iterations: ~{expected_iterations}")

    iteration = 0
    retrain_counter = 0
    model = None
    template_saved = False
    all_predictions = []
    all_upper = []
    all_lower = []
    start_time = time.time()

    while current_forecast_start <= test_end:
        iteration += 1

        # Slice the most recent N points as the training window
        train_window_start_idx = (
            df.index.get_loc(current_train_end) - train_window_length + 1
        )
        if train_window_start_idx < 0:
            train_window_start_idx = 0
        train_window_start = df.index[train_window_start_idx]
        train_window = df.loc[train_window_start:current_train_end]

        # Retrain the model at the start and then every `retrain_every` iterations
        if retrain_counter % retrain_every == 0:
            log.info(
                f"Iteration {iteration}: Training on {len(train_window)} points "
                f"({train_window.index[0].strftime('%Y-%m-%d')} to {train_window.index[-1].strftime('%Y-%m-%d')})"
            )

            # AutoTS configuration — see https://github.com/winedarksea/AutoTS#basic-use
            # "backwards" validation walks back from the most recent data
            # "no_negatives" clamps forecasts >= 0 (useful for energy load data)
            model_config = {
                "forecast_length": forecast_length_steps,
                "frequency": "infer",
                "prediction_interval": PREDICTION_INTERVAL,
                "ensemble": ENSEMBLE,
                "max_generations": MAX_GENERATIONS,
                "num_validations": NUM_VALIDATIONS,
                "validation_method": "backwards",
                "model_list": MODEL_LIST,
                "transformer_list": "fast",
                "drop_most_recent": 0,
                "n_jobs": N_JOBS,
                "verbose": 0,
                "no_negatives": True,
            }

            model = AutoTS(**model_config)

            # Import template BEFORE fit() — method="only" restricts search
            # to previously saved best models (faster convergence)
            if use_template and TEMPLATE_PATH.exists():
                try:
                    model = model.import_template(
                        str(TEMPLATE_PATH),
                        method="only",
                    )
                    log.info("  -> Template imported, using saved best models")
                except Exception as e:
                    log.warning(f"  Template import failed: {e}")

            # fit() runs the genetic algorithm search on the training window
            model = model.fit(train_window)

            # Export template after first successful training for future runs
            if not template_saved and not use_template:
                try:
                    model.export_template(
                        str(TEMPLATE_PATH),
                        models="best",
                        n=15,
                        max_per_model_class=3,
                    )
                    log.info(f"  -> Template exported: {TEMPLATE_PATH.name}")
                    log.info(f"  -> Best model: {model.best_model_name}")
                    template_saved = True
                    use_template = True
                except Exception as e:
                    log.warning(f"  Template export failed: {e}")
                    template_saved = True

        # Generate forecast using the current model
        if model is None:
            log.error(f"Model is None at iteration {iteration}")
            break

        # model.predict() returns a PredictionObject with .forecast,
        # .upper_forecast, and .lower_forecast DataFrames
        prediction = model.predict()
        forecast_df = prediction.forecast
        upper_df = prediction.upper_forecast
        lower_df = prediction.lower_forecast

        # Keep only the stride portion (avoids overlap with next forecast)
        forecast_end_idx = min(len(forecast_df), forecast_stride_steps)
        forecast_slice = forecast_df.iloc[:forecast_end_idx]
        upper_slice = upper_df.iloc[:forecast_end_idx]
        lower_slice = lower_df.iloc[:forecast_end_idx]

        if len(forecast_slice) > 0:
            all_predictions.append(forecast_slice[TARGET_COL])
            all_upper.append(upper_slice[TARGET_COL])
            all_lower.append(lower_slice[TARGET_COL])

        # Advance by stride (not full horizon) to avoid gaps
        current_forecast_start = current_forecast_start + pd.Timedelta(
            minutes=forecast_stride_steps * minutes_per_step
        )
        current_train_end = current_forecast_start - pd.Timedelta(
            minutes=minutes_per_step
        )
        retrain_counter += 1

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            progress = iteration / expected_iterations * 100
            if iteration > 0:
                eta_seconds = (elapsed / iteration) * (expected_iterations - iteration)
                eta_str = (
                    f", ETA: {eta_seconds/60:.0f} min"
                    if eta_seconds > 60
                    else f", ETA: {eta_seconds:.0f} sec"
                )
            else:
                eta_str = ""
            log.info(
                f"  → {iteration}/{expected_iterations} ({progress:.1f}%) bei {current_forecast_start.strftime('%Y-%m-%d %H:%M')}{eta_str}"
            )

    # Concatenate all forecast slices and deduplicate
    combined = pd.concat(all_predictions)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()

    combined_upper = pd.concat(all_upper)
    combined_upper = combined_upper[~combined_upper.index.duplicated(keep="first")]
    combined_upper = combined_upper.sort_index()

    combined_lower = pd.concat(all_lower)
    combined_lower = combined_lower[~combined_lower.index.duplicated(keep="first")]
    combined_lower = combined_lower.sort_index()

    log.info(f"Rolling window complete: {len(combined)} forecast points")

    return combined, combined_upper, combined_lower


# ============================ MAIN PIPELINE ============================
def run_pipeline():
    """Main entry point: load data, run rolling forecast, evaluate, save results."""
    log.info(f"=== Starting AutoTS Pipeline for {DATASET_NAME} ===")
    log.info(f"Horizon: {HORIZON_HOURS}h, Stride: {FORECAST_EVERY_HOURS}h")

    t0 = time.time()

    # 1) Load and prepare data
    df = load_and_prepare_data(CSV_PATH, TIMESTAMP_COL)

    forecast_year = FORECAST_YEAR
    log.info(f"Forecast year: {forecast_year}")

    # 2) Split: January = initial training, Feb-Dec = test
    training_start = pd.Timestamp(forecast_year, 1, 1)
    training_end = pd.Timestamp(forecast_year, 1, 31, 23, 0)
    test_start = pd.Timestamp(forecast_year, 2, 1)
    test_end = pd.Timestamp(forecast_year, 12, 31, 23, 0)

    # 3) Extract train/test slices
    january_data = df.loc[training_start:training_end]
    test_data = df.loc[test_start:test_end]

    if len(january_data) == 0:
        log.error("No January data available!")
        return

    if len(test_data) == 0:
        log.error("No test data (Feb-Dec) available!")
        return

    # 4) Derive steps per hour from actual data resolution
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    log.info(
        f"Training on Jan {forecast_year}: {len(january_data)} points "
        f"({len(january_data)/(steps_per_hour*24):.1f} days)"
    )
    log.info(
        f"Testing on Feb-Dec {forecast_year}: {len(test_data)} points "
        f"({len(test_data)/(steps_per_hour*24):.1f} days)"
    )

    # 5) Run rolling window forecast
    predictions, upper_predictions, lower_predictions = rolling_window_forecast(
        df=df,
        train_data=january_data,
        test_data=test_data,
        horizon_hours=HORIZON_HOURS,
        forecast_every_hours=FORECAST_EVERY_HOURS,
        steps_per_hour=steps_per_hour,
    )

    # 6) Evaluate on overlapping indices between test data and predictions
    common_idx = test_data.index.intersection(predictions.index)
    test_aligned = test_data.loc[common_idx, TARGET_COL]
    pred_aligned = predictions.loc[common_idx]
    upper_aligned = upper_predictions.loc[common_idx]
    lower_aligned = lower_predictions.loc[common_idx]

    if len(test_aligned) == 0:
        log.error("No overlap between test data and predictions!")
        return

    log.info(f"Evaluating over {len(test_aligned)} overlapping data points")
    metrics = compute_metrics(test_aligned, pred_aligned, upper_aligned, lower_aligned)
    log.info(
        f"Metrics: RMSE={metrics.rmse:.2f} kW, NRMSE={metrics.nrmse:.2f}%, "
        f"MAPE={metrics.mape:.2f}%, Pearson={metrics.pearson:.4f}, "
        f"Coverage={metrics.coverage:.1f}%, Interval width={metrics.interval_width_mean:.2f} kW"
    )

    # 7) Save results: plots (SVG) + metrics (JSON)
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    plot_path = (
        RESULTS_DIR
        / f"{DATASET_NAME}_autots_difference_{forecast_year}_{timestamp}.svg"
    )

    plot_simple_difference(
        test_aligned, pred_aligned, metrics, plot_path, forecast_year=forecast_year
    )

    comparison_plot_path = (
        RESULTS_DIR
        / f"{DATASET_NAME}_autots_comparison_{forecast_year}_{timestamp}.svg"
    )
    plot_comparison(
        test_aligned,
        pred_aligned,
        metrics,
        comparison_plot_path,
        forecast_year=forecast_year,
        upper_forecast=upper_aligned,
        lower_forecast=lower_aligned,
    )

    # 8) Save time series CSV (actual + predicted + prediction interval)
    timeseries_df = pd.DataFrame(
        {
            "timestamp": test_aligned.index,
            "actual": test_aligned.values,
            "predicted": pred_aligned.values,
            "lower_95": lower_aligned.values,
            "upper_95": upper_aligned.values,
        }
    )
    timeseries_df.set_index("timestamp", inplace=True)

    csv_path = (
        RESULTS_DIR
        / f"{DATASET_NAME}_timeseries_H{HORIZON_HOURS}h_{forecast_year}_{timestamp}.csv"
    )
    timeseries_df.to_csv(csv_path, encoding="utf-8")
    log.info(f"Time series saved: {csv_path} ({len(timeseries_df)} points)")

    results = {
        "dataset": DATASET_NAME,
        "forecast_year": forecast_year,
        "horizon_hours": HORIZON_HOURS,
        "forecast_every_hours": FORECAST_EVERY_HOURS,
        "metrics": {
            "rmse": metrics.rmse,
            "nrmse": metrics.nrmse,
            "mape": metrics.mape,
            "pearson": metrics.pearson,
            "coverage": metrics.coverage,
            "interval_width_mean": metrics.interval_width_mean,
        },
        "model_config": {
            "model_list": MODEL_LIST,
            "ensemble": ENSEMBLE,
            "max_generations": MAX_GENERATIONS,
            "num_validations": NUM_VALIDATIONS,
            "prediction_interval": PREDICTION_INTERVAL,
        },
    }

    json_path = (
        RESULTS_DIR
        / f"{DATASET_NAME}_results_H{HORIZON_HOURS}h_{forecast_year}_{timestamp}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Results saved to: {json_path}")
    log.info(f"Total runtime: {(time.time() - t0) / 60:.2f} minutes")


# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    run_pipeline()
