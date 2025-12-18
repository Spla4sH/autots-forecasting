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
CSV_NAME = "residential5.csv"  # Pfad zur CSV-Datei
FORECAST_YEAR = 2016

HORIZON_HOURS = 6  # Prognosehorizont in Stunden
FORECAST_EVERY_HOURS = 6  # Prognose-Intervall (wie oft neue Prognose)

N_JOBS = -1  # Windows hat Probleme mit -1

# AutoTS Modell-Konfiguration
MODEL_LIST = "default"  # default = alle robusten Modelle (fair für Vergleich mit optimiertem LightGBM)
ENSEMBLE = "simple"  # simple = Durchschnitt der besten Modelle (robust, funktioniert zuverlässig)
MAX_GENERATIONS = 5  # Template-System erlaubt höhere Werte (optimiert beste Modelle)
NUM_VALIDATIONS = 2  # Gute Balance zwischen Robustheit und Speed
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
TARGET_COL = "DE_KN_residential5_grid_import"

RESULTS_DIR = RESULTS_BASE_DIR / DATASET_NAME
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / CSV_NAME
TEMPLATE_PATH = RESULTS_DIR / f"{DATASET_NAME}_best_models.json"


# ============================ DATENSTRUKTUREN ============================
@dataclass
class Metrics:
    """Metriken für Forecast-Evaluation"""

    rmse: float
    nrmse: float
    mape: float
    pearson: float
    coverage: float  # % der Ist-Werte im 95% Konfidenzintervall
    interval_width_mean: float  # Durchschnittliche Breite des Konfidenzintervalls


# ============================ DATEN LADEN ============================
def load_and_prepare_data(csv_path: Path, time_col: str) -> pd.DataFrame:
    """CSV laden, als DataFrame vorbereiten, interpolieren."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    log.info(f"Lade Daten von: {csv_path}")

    df = pd.read_csv(csv_path, sep=";", usecols=[time_col, TARGET_COL])
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.drop_duplicates(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    # Konvertiere zu timezone-naive für AutoTS Kompatibilität
    df.index = df.index.tz_localize(None)

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
def compute_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    upper_forecast: pd.Series = None,
    lower_forecast: pd.Series = None,
) -> Metrics:
    """RMSE, NRMSE, MAPE, Pearson, Coverage, Intervall-Weite berechnen."""
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

    # Coverage & Intervall-Weite (falls Konfidenzintervall vorhanden)
    coverage_val = 0.0
    interval_width_mean = 0.0

    if upper_forecast is not None and lower_forecast is not None:
        upper_vals = upper_forecast.values.flatten()[:min_len]
        lower_vals = lower_forecast.values.flatten()[:min_len]

        # Coverage: Wie viel % der Ist-Werte liegen im Intervall?
        within_interval = (actual_vals >= lower_vals) & (actual_vals <= upper_vals)
        coverage_val = float(np.mean(within_interval) * 100)

        # Durchschnittliche Intervall-Weite
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
    upper_forecast: pd.Series = None,
    lower_forecast: pd.Series = None,
) -> None:
    """Vergleichsplot Ist vs. Prognose mit Konfidenzintervall."""
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))

    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

    # Align indices
    common_idx = val_series.index.intersection(prediction.index)
    val_aligned = val_series.loc[common_idx]
    pred_aligned = prediction.loc[common_idx]

    # Konfidenzintervall (falls vorhanden)
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

    # Optik
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#34495e")
    ax.spines["bottom"].set_color("#34495e")
    ax.tick_params(colors="#34495e", which="both")

    plt.tight_layout()
    fig.savefig(save_path, format="svg", dpi=300, bbox_inches="tight")
    log.info(f"Vergleichsplot gespeichert: {save_path}")
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

    retrain_every = (
        7 * 24 // forecast_every_hours
    )  # Wöchentlich (alle ~28 Iterationen bei 6h stride)

    # FESTES Rolling Window (wie LightGBM)
    train_window_length = len(train_data)  # Feste Länge = Januar
    test_start = test_data.index[0]
    test_end = test_data.index[-1]

    current_forecast_start = test_start
    current_train_end = train_data.index[-1]  # Initialer Train-End

    # Berechne forecast_length in Datenpunkten basierend auf tatsächlicher Frequenz
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    forecast_length_steps = horizon_hours * steps_per_hour
    forecast_stride_steps = forecast_every_hours * steps_per_hour

    log.info(f"Forecast-Länge: {horizon_hours}h = {forecast_length_steps} Datenpunkte")
    log.info(
        f"Forecast-Stride: {forecast_every_hours}h = {forecast_stride_steps} Datenpunkte"
    )
    log.info(
        f"Festes Rolling Window: {train_window_length} Punkte ({train_window_length/(steps_per_hour*24):.1f} Tage)"
    )
    log.info(f"Retraining alle {7 * 24 // forecast_every_hours} Iterationen (1 Woche)")

    # Template-basiertes Retraining (korrekte Methode aus AutoTS Dokumentation)
    use_template = TEMPLATE_PATH.exists()

    if use_template:
        log.info(f"Template gefunden: {TEMPLATE_PATH.name}")
    else:
        log.info("Kein Template - verwende initiale Modellsuche")

    # Berechne erwartete Iterationen für Fortschrittsanzeige
    test_duration_minutes = (test_end - test_start).total_seconds() / 60
    expected_iterations = int(test_duration_minutes / (forecast_every_hours * 60)) + 1
    log.info(f"Erwartete Iterationen: ~{expected_iterations}")

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

        # FESTES Rolling Window: Letzte N Punkte (wie LightGBM train_length)
        train_window_start_idx = (
            df.index.get_loc(current_train_end) - train_window_length + 1
        )
        if train_window_start_idx < 0:
            train_window_start_idx = 0
        train_window_start = df.index[train_window_start_idx]
        train_window = df.loc[train_window_start:current_train_end]

        # Retraining alle X Iterationen
        if retrain_counter % retrain_every == 0:
            log.info(
                f"Iteration {iteration}: Training auf {len(train_window)} Punkten "
                f"({train_window.index[0].strftime('%Y-%m-%d')} bis {train_window.index[-1].strftime('%Y-%m-%d')})"
            )

            # AutoTS-Konfiguration
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

            # Template importieren VOR fit() wenn vorhanden
            if use_template and TEMPLATE_PATH.exists():
                try:
                    model = model.import_template(
                        str(TEMPLATE_PATH),
                        method="only",  # Nur Template-Modelle verwenden
                    )
                    log.info("  → Template importiert, verwende beste Modelle")
                except Exception as e:
                    log.warning(f"  ⚠ Template-Import fehlgeschlagen: {e}")

            # Normales fit() - AutoTS optimiert die Template-Modelle neu
            model = model.fit(train_window)

            # Template speichern nach erstem erfolgreichen Training
            if not template_saved and not use_template:
                try:
                    model.export_template(
                        str(TEMPLATE_PATH),
                        models="best",
                        n=15,  # Top 15 Modelle exportieren
                        max_per_model_class=3,
                    )
                    log.info(f"  → Template exportiert: {TEMPLATE_PATH.name}")
                    log.info(f"  → Beste Modelle: {model.best_model_name}")
                    template_saved = True
                    use_template = True  # Für nächstes Retraining verwenden
                except Exception as e:
                    log.warning(f"  ⚠ Template-Export fehlgeschlagen: {e}")
                    template_saved = True

        # Prognose erstellen (falls model existiert)
        if model is None:
            log.error(f"Model ist None bei Iteration {iteration}")
            break

        prediction = model.predict()
        forecast_df = prediction.forecast  # .forecast gibt DataFrame zurück
        upper_df = prediction.upper_forecast
        lower_df = prediction.lower_forecast

        # Nur die nächsten forecast_stride_steps Datenpunkte speichern (stride)
        forecast_end_idx = min(len(forecast_df), forecast_stride_steps)
        forecast_slice = forecast_df.iloc[:forecast_end_idx]
        upper_slice = upper_df.iloc[:forecast_end_idx]
        lower_slice = lower_df.iloc[:forecast_end_idx]

        if len(forecast_slice) > 0:
            all_predictions.append(forecast_slice[TARGET_COL])
            all_upper.append(upper_slice[TARGET_COL])
            all_lower.append(lower_slice[TARGET_COL])

        # Nächste Iteration: Verschiebe um forecast_stride_steps (nicht um forecast_length!)
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

    # Alle Prognosen zusammenfügen
    combined = pd.concat(all_predictions)
    combined = combined[~combined.index.duplicated(keep="first")]  # Duplikate entfernen
    combined = combined.sort_index()

    combined_upper = pd.concat(all_upper)
    combined_upper = combined_upper[~combined_upper.index.duplicated(keep="first")]
    combined_upper = combined_upper.sort_index()

    combined_lower = pd.concat(all_lower)
    combined_lower = combined_lower[~combined_lower.index.duplicated(keep="first")]
    combined_lower = combined_lower.sort_index()

    log.info(f"Rolling Window abgeschlossen: {len(combined)} Prognosepunkte")

    return combined, combined_upper, combined_lower


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

    # 4) Berechne Schritte pro Stunde
    time_delta = df.index[1] - df.index[0]
    minutes_per_step = time_delta.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step)

    log.info(
        f"Training auf Januar {forecast_year}: {len(january_data)} Punkte "
        f"({len(january_data)/(steps_per_hour*24):.1f} Tage)"
    )
    log.info(
        f"Testing auf Feb-Dez {forecast_year}: {len(test_data)} Punkte "
        f"({len(test_data)/(steps_per_hour*24):.1f} Tage)"
    )

    # 5) Rolling Window Forecasting
    predictions, upper_predictions, lower_predictions = rolling_window_forecast(
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
    upper_aligned = upper_predictions.loc[common_idx]
    lower_aligned = lower_predictions.loc[common_idx]

    if len(test_aligned) == 0:
        log.error("Keine Überschneidung zwischen Test und Prognose!")
        return

    log.info(f"Evaluiere über {len(test_aligned)} überschneidende Datenpunkte")
    metrics = compute_metrics(test_aligned, pred_aligned, upper_aligned, lower_aligned)
    log.info(
        f"Metriken: RMSE={metrics.rmse:.2f} kW, NRMSE={metrics.nrmse:.2f}%, "
        f"MAPE={metrics.mape:.2f}%, Pearson={metrics.pearson:.4f}, "
        f"Coverage={metrics.coverage:.1f}%, Intervall-Weite={metrics.interval_width_mean:.2f} kW"
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

    # 8) Zeitreihen speichern (Ist + Prognose + Konfidenzintervall)
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
    log.info(f"Zeitreihen gespeichert: {csv_path} ({len(timeseries_df)} Punkte)")

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

    log.info(f"Ergebnisse gespeichert in: {json_path}")
    log.info(f"Gesamtlaufzeit: {(time.time() - t0) / 60:.2f} Minuten")


# ============================ ENTRY POINT ============================
if __name__ == "__main__":
    run_pipeline()
