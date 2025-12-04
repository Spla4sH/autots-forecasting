import warnings, logging, time, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.metrics import rmse, mape
from darts.dataprocessing.transformers import Scaler

# ============================ KONFIGURATION ============================
CSV_NAME = "building_0.csv"
FORECAST_YEAR = 2016

HORIZON_HOURS = 6  # Prognosehorizont
FORECAST_EVERY_HOURS = 6  # Prognose-Intervall

N_JOBS = -1  # Parallelisierung
DEBUG_MODE = True

# -------------------- HPO --------------------
ENABLE_HYPEROPT = True
N_TRIALS = 30
TIMEOUT_SECONDS = 1800
TUNING_OBJECTIVE = "RMSE"  # MAPE | NRMSE

USE_SAMPLE_WEIGHTS = False
WEIGHT_TYPE = "exponential"  # linear

# ============================ SETUP & LOGGING ============================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("lgbm-pipeline")

# Optuna Import
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None
    if ENABLE_HYPEROPT:
        log.warning("Optuna nicht installiert - HPO wird deaktiviert.")
        ENABLE_HYPEROPT = False

# ============================ PFADE ============================
DATASET_NAME = Path(CSV_NAME).stem
DATASET_ID = (
    DATASET_NAME.replace("residential", "R_cossmic")
    .replace("industry", "I_cossmic")
    .replace("Buero", "Buero_synGHD")
    .replace("Hochschule", "HS_SynGHD")
    .replace("Kultur", "Kult_synGHD")
    .replace("Sport", "Spo_synGHD")
    .replace("Technik", "Tech_synGHD")
    .replace("building", "B_synPRO")
)

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent
except NameError:
    PROJECT_ROOT = Path(".").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_BASE_DIR = PROJECT_ROOT / "results"
CSV_PATH = DATA_DIR / CSV_NAME

# CSV Spalten
TIMESTAMP_COL = "utc_timestamp"
TARGET_COL = "DE_KN_industrial1_load"

RESULTS_DIR = RESULTS_BASE_DIR / DATASET_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================ DATENSTRUKTUREN ============================
@dataclass
class Metrics:
    rmse: float
    nrmse: float
    mape: float
    pearson: float


# ============================ DATEN LADEN ============================
def load_and_prepare_data(csv_path: Path, time_col: str, value_col: str) -> TimeSeries:
    """CSV laden, TimeSeries erstellen, interpolieren."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    log.info(f"Lade Daten von: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", usecols=[time_col, value_col])
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)

    ts = TimeSeries.from_dataframe(
        df,
        time_col=time_col,
        value_cols=value_col,
        fill_missing_dates=True,
        fillna_value=None,
    )

    # Lineare Interpolation + Randbehandlung
    df_interp = ts.to_dataframe().interpolate("linear").bfill().ffill()
    ts = TimeSeries.from_dataframe(df_interp)

    freq = ts.freq
    if freq:
        log.info(f"Erkannte Frequenz: {freq}")
    else:
        log.warning("Frequenz konnte nicht automatisch erkannt werden.")

    log.info(
        f"Zeitreihe geladen: {len(ts)} Punkte, {ts.start_time()} bis {ts.end_time()}"
    )
    return ts


# ============================ METRIKEN ============================
def compute_metrics(actual: TimeSeries, pred: TimeSeries) -> Metrics:
    """RMSE, NRMSE, MAPE, Pearson berechnen."""
    actual_vals = actual.values().flatten()
    pred_vals = pred.values().flatten()

    min_len = min(len(actual_vals), len(pred_vals))
    if min_len == 0:
        raise ValueError("Keine überlappenden Daten")

    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]
    if len(actual) > min_len:
        actual = actual[:min_len]
    if len(pred) > min_len:
        pred = pred[:min_len]

    rmse_val = rmse(actual, pred)

    # NRMSE
    mean_val = float(np.mean(actual_vals))
    nrmse_val = (
        (float(rmse_val / mean_val) * 100) if abs(mean_val) > 1e-6 else float("inf")
    )

    # MAPE mit Fallback
    try:
        mape_val = mape(actual, pred)
    except Exception:
        eps = 1e-6
        mape_val = float(
            np.mean(
                np.abs((actual_vals - pred_vals) / np.maximum(np.abs(actual_vals), eps))
            )
            * 100
        )

    # Pearson
    try:
        from scipy.stats import pearsonr

        pearson_val = (
            float(pearsonr(actual_vals, pred_vals)[0])
            if len(actual_vals) > 1
            else float("nan")
        )
    except Exception:
        pearson_val = (
            float(np.corrcoef(actual_vals, pred_vals)[0, 1])
            if len(actual_vals) > 1
            else float("nan")
        )

    return Metrics(
        rmse=float(rmse_val),
        nrmse=float(nrmse_val),
        mape=float(mape_val),
        pearson=float(pearson_val),
    )


# ============================ PLOTTING ============================
def plot_simple_difference(
    val_series: TimeSeries,
    prediction: TimeSeries,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int = 2016,
) -> None:
    """Fehler-Plot (Prognose - Ist)."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    min_len = min(len(val_series), len(prediction))
    val_trimmed = val_series[:min_len]
    pred_trimmed = prediction[:min_len]

    diff = pred_trimmed.values() - val_trimmed.values()

    ax.plot(
        pred_trimmed.time_index,
        diff.flatten(),
        color="darkblue",
        linewidth=0.5,
        alpha=0.7,
    )
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1.0)

    ax.set_title(
        f"Vorhersagefehler {forecast_year} (native Auflösung) - RMSE: {metrics.rmse:.2f} kW | "
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
    log.info(f"Differenzplot gespeichert: {save_path}")
    plt.close(fig)


def plot_comparison(
    val_series: TimeSeries,
    prediction: TimeSeries,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int = 2016,
) -> None:
    """Vergleichsplot Ist vs. Prognose (Stundenmittel für Anzeige)."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # Stundenmittel für Anzeige
    val_hourly = val_series.resample("1H", method="mean")
    pred_hourly = prediction.resample("1H", method="mean")

    ax.plot(
        val_hourly.time_index,
        val_hourly.values().flatten(),
        color="#1f4e79",
        linewidth=1.0,
        alpha=0.95,
        label="Tatsächliche Last (Stundenmittel)",
        zorder=3,
    )
    ax.plot(
        pred_hourly.time_index,
        pred_hourly.values().flatten(),
        color="#d62728",
        linewidth=1.0,
        alpha=0.95,
        label="LightGBM Prognose (Stundenmittel)",
        zorder=2,
    )

    ax.set_title(
        f"LightGBM Jahresprognose ({CSV_NAME}) {forecast_year} Vorhersagehorizont: {HORIZON_HOURS}h - "
        f"RMSE: {metrics.rmse:.2f} kW | NRMSE: {metrics.nrmse:.1f}% | "
        f"MAPE: {metrics.mape:.1f}% | Pearson: {metrics.pearson:.3f}",
        fontsize=18,
        fontweight="bold",
        pad=25,
    )
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

    ax.set_xlim(val_hourly.time_index[0], val_hourly.time_index[-1])
    ax.set_ylim(bottom=0)
    ax.margins(x=0, y=0.02)

    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5, color="#bdc3c7")
    ax.grid(
        True, alpha=0.2, linestyle="-", linewidth=0.3, color="#bdc3c7", which="minor"
    )

    # Optik
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#34495e")
    ax.spines["bottom"].set_color("#34495e")
    ax.tick_params(colors="#34495e", which="both")

    plt.tight_layout()

    comparison_path = save_path.with_name(
        save_path.name.replace("_difference", "_comparison")
    )
    fig.savefig(comparison_path, format="svg", dpi=300, bbox_inches="tight")
    log.info(f"Vergleichsplot gespeichert: {comparison_path}")
    plt.close(fig)


# ============================ HPO ============================
def run_optuna_optimization(
    series: TimeSeries, horizon_hours: int, forecast_year: int
) -> Optional[Dict[str, Any]]:
    """HPO auf Januar: 80/20 Split für Hyperparameter-Validierung."""
    if not ENABLE_HYPEROPT or optuna is None:
        return None

    study_name = f"lgbm_{DATASET_ID}_H{horizon_hours}h_Y{forecast_year}"
    log.info(f"Starte Optuna-Optimierung: {study_name}")

    # 80/20 Split
    hpo_series = series
    split_point = int(len(hpo_series) * 0.8)
    train_hpo = hpo_series[:split_point]
    val_hpo = hpo_series[split_point:]

    log.info(
        f"HPO Training:   {train_hpo.start_time()} - {train_hpo.end_time()} ({len(train_hpo)} Punkte)"
    )
    log.info(
        f"HPO Validierung: {val_hpo.start_time()} - {val_hpo.end_time()} ({len(val_hpo)} Punkte)"
    )

    if train_hpo.freq != val_hpo.freq:
        log.warning(f"Frequenz-Mismatch: Train={train_hpo.freq}, Val={val_hpo.freq}")

    min_val_length = 24 * 4  # Min 1 Tag
    if len(val_hpo) < min_val_length:
        log.warning("Validierung zu kurz - Aufteilung wird angepasst.")
        split_point = len(hpo_series) - min_val_length
        train_hpo = hpo_series[:split_point]
        val_hpo = hpo_series[split_point:]

    def objective(trial) -> float:
        # Frequenz → Schritte
        time_delta = train_hpo.time_index[1] - train_hpo.time_index[0]
        minutes_per_step = time_delta.total_seconds() / 60
        steps_per_hour = int(60 / minutes_per_step)
        steps_per_day = steps_per_hour * 24

        # Lags: 1-2 Tage
        lags = trial.suggest_int("lags", steps_per_day * 1, steps_per_day * 2)

        horizon_steps = int(horizon_hours * steps_per_hour)
        output_chunk_length = horizon_steps
        multi_models = True

        # LightGBM Parameter
        n_estimators = trial.suggest_int("n_estimators", 200, 1200)
        num_leaves = trial.suggest_int("num_leaves", 31, 255)
        min_child_samples = trial.suggest_int("min_child_samples", 10, 200)

        learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True)
        max_depth = trial.suggest_int("max_depth", -1, 12)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0, 2.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0, 10.0)

        # Past Covariates
        past_lag_depth = trial.suggest_int("past_lag_depth", 1, 3)
        max_back_hours = 12

        min_lag = -steps_per_hour * max_back_hours
        past_cov_lags = sorted(
            [
                trial.suggest_int(f"past_lag_{i}", min_lag, -1)
                for i in range(past_lag_depth)
            ]
        )

        # Future Covariates: ganzer Horizont
        future_cov_lags = list(range(0, output_chunk_length))

        # Encoder
        encoders_hpo = {
            "cyclic": {
                "past": ["hour", "dayofweek", "month"],
                "future": ["hour", "dayofweek", "month"],
            },
            "datetime_attribute": {"past": ["dayofyear"], "future": ["dayofyear"]},
            "transformer": Scaler(),
        }

        # Modell fitten
        model_params_hpo = {
            "lags": lags,
            "lags_past_covariates": past_cov_lags,
            "lags_future_covariates": future_cov_lags,
            "output_chunk_length": output_chunk_length,
            "multi_models": multi_models,
            "add_encoders": encoders_hpo,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": 42,
            "n_jobs": 3,
            "verbose": -1,
        }

        try:
            model = LightGBMModel(**model_params_hpo)
            model.fit(series=train_hpo)

            # Validierung
            forecasts = model.historical_forecasts(
                series=hpo_series,
                start=val_hpo.start_time(),
                forecast_horizon=output_chunk_length,
                stride=output_chunk_length,
                retrain=False,
                last_points_only=False,
                show_warnings=False,
                enable_optimization=True,
            )

            # Chunks zusammenfügen
            if isinstance(forecasts, list):
                pred = forecasts[0]
                for f in forecasts[1:]:
                    pred = pred.concatenate(f, axis=0)
            else:
                pred = forecasts

            # Metrik berechnen
            pred = pred.slice_intersect(val_hpo)

            if TUNING_OBJECTIVE == "RMSE":
                return float(rmse(val_hpo, pred))
            elif TUNING_OBJECTIVE == "MAPE":
                return float(mape(val_hpo, pred))
            elif TUNING_OBJECTIVE == "NRMSE":
                rmse_val = rmse(val_hpo, pred)
                mean_val = float(np.mean(val_hpo.values()))
                return (
                    float(rmse_val / mean_val) if abs(mean_val) > 1e-6 else float("inf")
                )
            else:
                return float(rmse(val_hpo, pred))

        except Exception as e:
            log.debug(f"Trial fehlgeschlagen: {e}")
            return float("inf")

    # Optuna Studie
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        n_jobs=1,
        show_progress_bar=True,
    )

    log.info(f"Beste Hyperparameter: {study.best_params}")
    log.info(f"Bester Score: {study.best_value:.6f}")

    # Trials speichern
    trials_df = study.trials_dataframe()
    trials_csv = RESULTS_DIR / f"optuna_trials_{study_name}.csv"
    trials_df.to_csv(trials_csv, index=False)

    return study.best_params


# ============================ MAIN PIPELINE ============================
def run_pipeline():
    log.info(f"=== Starte Pipeline für {DATASET_NAME} ===")
    log.info(f"Horizont: {HORIZON_HOURS}h, Stride: {FORECAST_EVERY_HOURS}h")

    t0 = time.time()

    # 1) Daten laden
    try:
        series = load_and_prepare_data(CSV_PATH, TIMESTAMP_COL, TARGET_COL)
    except Exception as e:
        log.error(f"Fehler beim Laden der Daten: {e}")
        return

    tz = series.start_time().tzinfo
    forecast_year = FORECAST_YEAR
    log.info(f"Prognose-Jahr: {forecast_year}")

    # 2) Zeitfenster (Jan=Training, Feb-Dez=Test)
    forecast_start = pd.Timestamp(forecast_year, 1, 1, tz=tz)
    forecast_end = pd.Timestamp(forecast_year, 12, 31, 23, 45, tz=tz)
    training_start = pd.Timestamp(forecast_year, 1, 1, tz=tz)
    training_end = pd.Timestamp(forecast_year, 1, 31, 23, 45, tz=tz)
    test_start = pd.Timestamp(forecast_year, 2, 1, tz=tz)
    test_end = pd.Timestamp(forecast_year, 12, 31, 23, 45, tz=tz)

    # 3) Frequenz
    time_delta = series.time_index[1] - series.time_index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    # 4) Daten extrahieren
    year_series = series.slice(forecast_start, forecast_end)
    if len(year_series) == 0:
        log.error(f"Keine Daten für Jahr {forecast_year} gefunden!")
        log.error(
            f"Verfügbare Daten: {series.start_time().year} bis {series.end_time().year}"
        )
        return

    january_series = year_series.slice(training_start, training_end)
    if len(january_series) == 0:
        log.error("Keine Januar-Daten verfügbar!")
        return

    test_series = year_series.slice(test_start, test_end)
    if len(test_series) == 0:
        log.error("Keine Test-Daten (Feb-Dez) verfügbar!")
        return

    log.info(
        f"Training auf Januar {forecast_year}: {len(january_series)} Punkte "
        f"({len(january_series)/(steps_per_hour*24):.1f} Tage)"
    )
    log.info(
        f"Testing auf Feb-Dez {forecast_year}: {len(test_series)} Punkte "
        f"({len(test_series)/(steps_per_hour*24):.1f} Tage)"
    )

    # Für Rolling Forecast wird komplette Jahresserie verwendet (Training+Test)
    full_series = year_series
    min_train_points = len(january_series)  # Rolling-Fensterlänge = Januar
    actual_start = test_start  # Rolling Forecast startet am 1. Februar

    # 5) HPO auf Januar-Daten (falls aktiviert)
    if ENABLE_HYPEROPT:
        log.info(
            f"Führe HPO auf Januar {forecast_year} durch ({len(january_series)} Punkte)"
        )
        best_params = run_optuna_optimization(
            january_series, HORIZON_HOURS, forecast_year
        )
    else:
        best_params = None

    # 6) Encoder immer aktiv (past & future); Future-Covariates decken Horizont ab
    output_chunk_length_global = int(HORIZON_HOURS * steps_per_hour)
    encoders = {
        "cyclic": {
            "past": ["hour", "dayofweek", "month"],
            "future": ["hour", "dayofweek", "month"],
        },
        "datetime_attribute": {"past": ["dayofyear"], "future": ["dayofyear"]},
        "position": {"past": ["relative"], "future": ["relative"]},
        "transformer": Scaler(),
    }

    # 7) Finale Modellparameter aus HPO übernehmen (oder Fallbacks nutzen)
    if best_params:
        # Past-Lags aus HPO rekonstruieren
        past_lag_depth = int(best_params.get("past_lag_depth", 3))
        if past_lag_depth > 0:
            past_cov_lags = []
            for i in range(past_lag_depth):
                lag = best_params.get(f"past_lag_{i}")
                if lag is not None:
                    past_cov_lags.append(int(lag))
            past_cov_lags = (
                sorted(past_cov_lags) if past_cov_lags else [-24, -12, -6, -3, -1]
            )
        else:
            past_cov_lags = [-24, -12, -6, -3, -1]

        # Standard Future-Lags = ganzer Horizont
        future_cov_lags = list(range(0, output_chunk_length_global))

        # Alle relevanten LightGBM-Parameter und Lags setzen
        model_params = {
            "lags": int(best_params.get("lags", steps_per_hour * 24)),
            "output_chunk_length": output_chunk_length_global,
            "multi_models": best_params.get("multi_models", True),
            "lags_past_covariates": past_cov_lags,
            "lags_future_covariates": future_cov_lags,
            "add_encoders": encoders,
            "n_estimators": int(best_params.get("n_estimators", 400)),
            "max_depth": int(best_params.get("max_depth", -1)),
            "num_leaves": int(best_params.get("num_leaves", 63)),
            "learning_rate": float(best_params.get("learning_rate", 0.05)),
            "min_child_samples": int(best_params.get("min_child_samples", 20)),
            "subsample": float(best_params.get("subsample", 0.9)),
            "colsample_bytree": float(best_params.get("colsample_bytree", 0.9)),
            "reg_alpha": float(best_params.get("reg_alpha", 0.01)),
            "reg_lambda": float(best_params.get("reg_lambda", 0.01)),
        }
    else:
        # Fallback: robuste Standardwerte, falls keine HPO lief
        past_cov_lags = [-24, -12, -6, -3, -1]
        future_cov_lags = list(range(0, output_chunk_length_global))
        model_params = {
            "lags": steps_per_hour * 24 * 2,
            "output_chunk_length": output_chunk_length_global,
            "multi_models": True,
            "lags_past_covariates": past_cov_lags,
            "lags_future_covariates": future_cov_lags,
            "add_encoders": encoders,
            "n_estimators": 400,
            "max_depth": -1,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
        }

    # Reproduzierbarkeit/Parallelisierung
    model_params.update({"random_state": 42, "n_jobs": N_JOBS, "verbose": -1})

    # Modell erstellen
    model = LightGBMModel(**model_params)
    log.info("LightGBM Modell erstellt für Rolling Window Forecasting")

    # 8) Rolling-Window-Backtest: Prognose-Chunks über das Jahr erzeugen
    #    - horizon = HORIZON_HOURS * steps_per_hour
    #    - stride  = FORECAST_EVERY_HOURS * steps_per_hour
    #    - retrain = etwa wöchentlich (als Intervall in 'Anzahl Starts')
    steps_per_horizon = int(HORIZON_HOURS * steps_per_hour)
    stride_in_points = int(FORECAST_EVERY_HOURS * steps_per_hour)
    steps_per_week = int(steps_per_hour * 24 * 7)
    steps_per_stride = stride_in_points
    retrain_interval = steps_per_week // steps_per_stride if steps_per_stride > 0 else 1

    if DEBUG_MODE:
        log.info(
            f"Erkannte Datenfrequenz: {minutes_per_step:.1f} Minuten pro Datenpunkt"
        )
        log.info(f"Datenpunkte pro Stunde: {steps_per_hour:.0f}")
        log.info(f"Datenpunkte pro Horizont (H): {steps_per_horizon}")
        log.info(
            f"Stride in Punkten: {stride_in_points} (= {FORECAST_EVERY_HOURS} Stunden)"
        )
        log.info(f"Train-Length (Rolling Window): {min_train_points} Punkte")
        expected_forecasts = len(test_series) / steps_per_stride
        log.info(f"Erwartete Anzahl Prognosen: {expected_forecasts:.0f}")
        log.info(f"Retraining alle {retrain_interval} Iterationen (1 Woche)")

    # Historische Prognosen erzeugen (Darts Backtesting)
    try:
        sample_weight_param = WEIGHT_TYPE if USE_SAMPLE_WEIGHTS else None

        historical_forecasts = model.historical_forecasts(
            series=full_series,
            start=actual_start,
            forecast_horizon=steps_per_horizon,
            stride=stride_in_points,
            retrain=retrain_interval,  # alle 'retrain_interval' Starts neu fitten
            train_length=min_train_points,  # Rolling-Fensterlänge = Januar
            last_points_only=False,  # alle Punkte des Chunks behalten
            enable_optimization=True,
            verbose=True,
            show_warnings=False,
            num_samples=1,
            sample_weight=sample_weight_param,
        )

        # Liste von Prognose-Chunks zu einer Serie zusammenfügen (falls nötig)
        if isinstance(historical_forecasts, list):
            combined_forecast = historical_forecasts[0]
            for forecast in historical_forecasts[1:]:
                combined_forecast = combined_forecast.concatenate(forecast, axis=0)
        else:
            combined_forecast = historical_forecasts

        log.info(
            f"Historical Forecasts generiert: {len(combined_forecast)} Datenpunkte"
        )
        log.info(
            f"Prognose-Zeitraum: {combined_forecast.start_time()} bis {combined_forecast.end_time()}"
        )

    except Exception as e:
        log.error(f"Fehler bei Historical Forecasts: {e}")
        raise e

    # 9) Evaluation nur auf Überschneidung zwischen Testfenster (Feb-Dez) und Prognoseserie
    overlap_start = max(test_series.start_time(), combined_forecast.start_time())
    overlap_end = min(test_series.end_time(), combined_forecast.end_time())

    test_overlap = test_series.slice(overlap_start, overlap_end)
    forecast_overlap = combined_forecast.slice(overlap_start, overlap_end)

    if len(test_overlap) == 0 or len(forecast_overlap) == 0:
        log.error("Keine Überschneidung zwischen Test und Prognose!")
        return

    log.info(f"Evaluiere über {len(test_overlap)} überschneidende Datenpunkte")
    metrics = compute_metrics(test_overlap, forecast_overlap)
    log.info(
        f"Metriken: RMSE={metrics.rmse:.2f} kW, NRMSE={metrics.nrmse:.2f}%, "
        f"MAPE={metrics.mape:.2f}%, Pearson={metrics.pearson:.4f}"
    )

    # 10) Ergebnisse (Plots + JSON) speichern
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    plot_path = (
        RESULTS_DIR / f"{DATASET_ID}_lgbm_difference_{forecast_year}_{timestamp}.svg"
    )

    plot_simple_difference(
        test_overlap, forecast_overlap, metrics, plot_path, forecast_year=forecast_year
    )
    plot_comparison(
        test_overlap, forecast_overlap, metrics, plot_path, forecast_year=forecast_year
    )

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
        },
        "model_params": model_params,
        "hpo_enabled": ENABLE_HYPEROPT,
        "sample_weights": USE_SAMPLE_WEIGHTS,
        "weight_type": WEIGHT_TYPE if USE_SAMPLE_WEIGHTS else None,
    }

    json_path = (
        RESULTS_DIR
        / f"{DATASET_ID}_results_H{HORIZON_HOURS}h_{forecast_year}_{timestamp}.json"
    )
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Ergebnisse gespeichert in: {json_path}")
    log.info(f"Gesamtlaufzeit: {(time.time() - t0) / 60:.2f} Minuten")


# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    if not DATA_DIR.exists():
        log.error(f"Datenverzeichnis nicht gefunden: {DATA_DIR}")
    else:
        run_pipeline()
