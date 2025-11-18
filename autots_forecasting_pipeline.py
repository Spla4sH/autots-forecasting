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

# ============================ KONFIGURATION ============================
CSV_NAME = "building_0.csv"  # Pfad zur CSV-Datei
FORECAST_YEAR = 2016

HORIZON_HOURS = 24  # Prognosehorizont in Stunden
FORECAST_EVERY_HOURS = 24  # Prognose-Intervall (wie oft neue Prognose)

N_JOBS = 1  # AutoTS Parallelisierung (1 für Windows-Stabilität)

# AutoTS Modell-Konfiguration
MODEL_LIST = "superfast"  # superfast | fast | all
ENSEMBLE = "simple"  # simple | distance | horizontal
MAX_GENERATIONS = 3
NUM_VALIDATIONS = 2
PREDICTION_INTERVAL = 0.95

# ============================ SETUP & LOGGING ============================
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("autots-pipeline")

# ============================ PFADE ============================
DATASET_NAME = Path(CSV_NAME).stem

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR
except NameError:
    PROJECT_ROOT = Path(".").resolve()

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_BASE_DIR = PROJECT_ROOT / "results"

# CSV Spalten
TIMESTAMP_COL = "utc_timestamp"
TARGET_COL = "DE_KN_industrial1_load"

RESULTS_DIR = RESULTS_BASE_DIR / DATASET_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / CSV_NAME


# ============================ DATENSTRUKTUREN ============================
@dataclass
class Metrics:
    """Metriken für Forecast-Evaluation"""

    rmse: float
    nrmse: float
    mape: float
    pearson: float


# ============================ DATEN LADEN ============================
def load_and_prepare_data(csv_path: Path, time_col: str) -> pd.DataFrame:
    """CSV laden, als DataFrame vorbereiten, interpolieren."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    log.info(f"Lade Daten von: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", usecols=[time_col, TARGET_COL])
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    # Lineare Interpolation
    df = df.interpolate(method="linear").bfill().ffill()

    freq = pd.infer_freq(df.index)
    if freq:
        log.info(f"Erkannte Frequenz: {freq}")
    else:
        log.warning("Frequenz konnte nicht automatisch erkannt werden")

    # Berechne Schritte pro Stunde für korrekte forecast_length Berechnung
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    log.info(f"Daten geladen: {len(df)} Punkte, {df.index[0]} bis {df.index[-1]}")
    log.info(
        f"Zeitauflösung: {minutes_per_step:.1f} Minuten pro Schritt ({steps_per_hour} Schritte/Stunde)"
    )

    return df


# ============================ METRIKEN ============================
def compute_metrics(actual: pd.Series, predicted: pd.Series) -> Metrics:
    """RMSE, NRMSE, MAPE, Pearson berechnen."""
    actual_vals = actual.values.flatten()
    pred_vals = predicted.values.flatten()

    min_len = min(len(actual_vals), len(pred_vals))
    if min_len == 0:
        raise ValueError("Keine überlappenden Daten")

    actual_vals = actual_vals[:min_len]
    pred_vals = pred_vals[:min_len]

    # RMSE
    rmse_val = np.sqrt(np.mean((actual_vals - pred_vals) ** 2))

    # NRMSE (normalisiert durch Mittelwert)
    mean_val = float(np.mean(actual_vals))
    nrmse_val = (rmse_val / mean_val * 100) if abs(mean_val) > 1e-6 else float("inf")

    # MAPE
    eps = 1e-6
    mape_val = float(
        np.mean(
            np.abs((actual_vals - pred_vals) / np.maximum(np.abs(actual_vals), eps))
        )
        * 100
    )

    # Pearson
    pearson_val = (
        float(pearsonr(actual_vals, pred_vals)[0])
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
    val_series: pd.Series,
    prediction: pd.Series,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int,
) -> None:
    """Fehler-Plot (Prognose - Ist)."""
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
    log.info(f"Differenzplot gespeichert: {save_path}")
    plt.close(fig)


def plot_comparison(
    val_series: pd.Series,
    prediction: pd.Series,
    metrics: Metrics,
    save_path: Path,
    forecast_year: int,
) -> None:
    """Vergleichsplot Ist vs. Prognose."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # Align indices
    common_idx = val_series.index.intersection(prediction.index)
    val_aligned = val_series.loc[common_idx]
    pred_aligned = prediction.loc[common_idx]

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

    ax.set_title(
        f"AutoTS Jahresprognose ({CSV_NAME}) {forecast_year} - Horizont: {HORIZON_HOURS}h - "
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

    ax.set_xlim(val_aligned.index[0], val_aligned.index[-1])
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


# ============================ ROLLING WINDOW FORECASTING ============================
def rolling_window_forecast(
    df: pd.DataFrame,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    horizon_hours: int,
    forecast_every_hours: int,
    steps_per_hour: int,
) -> pd.Series:
    """
    Rolling Window Forecasting mit wöchentlichem Retraining.

    Args:
        df: Gesamter Datensatz (für rolling window)
        train_data: Initiale Trainingsdaten (Januar)
        test_data: Testdaten (Feb-Dez)
        horizon_hours: Forecast-Horizont in Stunden
        forecast_every_hours: Wie oft eine neue Prognose erstellt wird

    Returns:
        Series mit allen Prognosen
    """
    log.info("=== Starte Rolling Window Forecasting ===")

    all_predictions = []
    retrain_counter = 0
    retrain_every = (
        7 * 24 // forecast_every_hours
    )  # Wöchentlich (alle ~7 Iterationen bei 24h stride)

    current_train_end = train_data.index[-1]
    test_start = test_data.index[0]
    test_end = test_data.index[-1]

    current_forecast_start = test_start

    # Berechne forecast_length in Datenpunkten basierend auf tatsächlicher Frequenz
    forecast_length_steps = horizon_hours * steps_per_hour
    forecast_stride_steps = forecast_every_hours * steps_per_hour

    log.info(f"Forecast-Länge: {horizon_hours}h = {forecast_length_steps} Datenpunkte")
    log.info(
        f"Forecast-Stride: {forecast_every_hours}h = {forecast_stride_steps} Datenpunkte"
    )

    # AutoTS Modell einmalig konfigurieren
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
    }

    iteration = 0

    while current_forecast_start <= test_end:
        iteration += 1

        # Trainings-Fenster: Vom Anfang bis aktuelles Ende
        train_window = df.loc[:current_train_end]

        # Retraining alle X Iterationen
        if retrain_counter % retrain_every == 0:
            log.info(
                f"Iteration {iteration}: Training auf {len(train_window)} Punkten "
                f"({train_window.index[0].strftime('%Y-%m-%d')} bis {train_window.index[-1].strftime('%Y-%m-%d')})"
            )

            model = AutoTS(**model_config)
            model = model.fit(train_window)

        # Prognose erstellen
        prediction = model.predict()
        forecast_df = prediction.forecast

        # Nur die nächsten forecast_stride_steps Datenpunkte speichern (stride)
        forecast_end_idx = min(len(forecast_df), forecast_stride_steps)
        forecast_slice = forecast_df.iloc[:forecast_end_idx]

        if len(forecast_slice) > 0:
            all_predictions.append(forecast_slice[TARGET_COL])

        # Nächste Iteration
        if len(forecast_slice) > 0:
            current_forecast_start = forecast_slice.index[-1] + (
                forecast_slice.index[1] - forecast_slice.index[0]
            )
            current_train_end = current_forecast_start - (df.index[1] - df.index[0])
        else:
            break
        retrain_counter += 1

        if iteration % 10 == 0:
            log.info(
                f"  → {iteration} Prognosen erstellt, aktuell bei {current_forecast_start.strftime('%Y-%m-%d')}"
            )

    # Alle Prognosen zusammenfügen
    combined = pd.concat(all_predictions)
    combined = combined[~combined.index.duplicated(keep="first")]  # Duplikate entfernen
    combined = combined.sort_index()

    log.info(f"Rolling Window abgeschlossen: {len(combined)} Prognosepunkte")

    return combined


# ============================ MAIN PIPELINE ============================
def run_pipeline():
    log.info(f"=== Starte AutoTS Pipeline für {DATASET_NAME} ===")
    log.info(f"Horizont: {HORIZON_HOURS}h, Stride: {FORECAST_EVERY_HOURS}h")

    t0 = time.time()

    # 1) Daten laden
    df = load_and_prepare_data(CSV_PATH, TIMESTAMP_COL)

    forecast_year = FORECAST_YEAR
    log.info(f"Prognose-Jahr: {forecast_year}")

    # 2) Zeitfenster (Jan=Training, Feb-Dez=Test)
    training_start = pd.Timestamp(forecast_year, 1, 1)
    training_end = pd.Timestamp(forecast_year, 1, 31, 23, 0)
    test_start = pd.Timestamp(forecast_year, 2, 1)
    test_end = pd.Timestamp(forecast_year, 12, 31, 23, 0)

    # 3) Daten extrahieren
    january_data = df.loc[training_start:training_end]
    test_data = df.loc[test_start:test_end]

    if len(january_data) == 0:
        log.error("Keine Januar-Daten verfügbar!")
        return

    if len(test_data) == 0:
        log.error("Keine Test-Daten (Feb-Dez) verfügbar!")
        return

    log.info(
        f"Training auf Januar {forecast_year}: {len(january_data)} Punkte "
        f"({len(january_data)/24:.1f} Tage)"
    )
    log.info(
        f"Testing auf Feb-Dez {forecast_year}: {len(test_data)} Punkte "
        f"({len(test_data)/24:.1f} Tage)"
    )

    # 4) Berechne Schritte pro Stunde
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    # 5) Rolling Window Forecasting
    predictions = rolling_window_forecast(
        df=df,
        train_data=january_data,
        test_data=test_data,
        horizon_hours=HORIZON_HOURS,
        forecast_every_hours=FORECAST_EVERY_HOURS,
        steps_per_hour=steps_per_hour,
    )

    # 6) Evaluation auf Überschneidung
    common_idx = test_data.index.intersection(predictions.index)
    test_aligned = test_data.loc[common_idx, TARGET_COL]
    pred_aligned = predictions.loc[common_idx]

    if len(test_aligned) == 0:
        log.error("Keine Überschneidung zwischen Test und Prognose!")
        return

    log.info(f"Evaluiere über {len(test_aligned)} überschneidende Datenpunkte")
    metrics = compute_metrics(test_aligned, pred_aligned)
    log.info(
        f"Metriken: RMSE={metrics.rmse:.2f} kW, NRMSE={metrics.nrmse:.2f}%, "
        f"MAPE={metrics.mape:.2f}%, Pearson={metrics.pearson:.4f}"
    )

    # 7) Ergebnisse (Plots + JSON) speichern
    timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    plot_path = (
        RESULTS_DIR
        / f"{DATASET_NAME}_autots_difference_{forecast_year}_{timestamp}.svg"
    )

    plot_simple_difference(
        test_aligned, pred_aligned, metrics, plot_path, forecast_year=forecast_year
    )
    plot_comparison(
        test_aligned, pred_aligned, metrics, plot_path, forecast_year=forecast_year
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
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"Ergebnisse gespeichert in: {json_path}")
    log.info(f"Gesamtlaufzeit: {(time.time() - t0) / 60:.2f} Minuten")


# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    run_pipeline()
