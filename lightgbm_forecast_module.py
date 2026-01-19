import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

from darts import TimeSeries
from darts.models import LightGBMModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import rmse, mape

# Optuna Import (optional)
try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("lgbm-module")


def make_forecast(
    load: list,
    unixtime: list,
    horizon: int,
    col_name: str = "load",
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Erstellt eine LightGBM-basierte Prognose.

    Args:
        load: Liste mit Lastwerten (kW)
        unixtime: Liste mit Unix-Timestamps (Sekunden)
        horizon: Anzahl der Prognoseschritte (in der Frequenz der Eingabedaten)
        col_name: Spaltenname für die Zeitreihe
        model_params: Optionale LightGBM-Parameter (überschreiben Defaults)

    Returns:
        DataFrame mit Prognose, Index ist DatetimeIndex (Europe/Berlin)
    """
    # 1) Trainingsdaten als DataFrame mit DatetimeIndex erstellen
    datetimeindex_train = pd.to_datetime(unixtime, unit="s", utc=True)
    datetimeindex_train = datetimeindex_train.tz_convert("Europe/Berlin")
    train_df = pd.DataFrame(data=load, index=datetimeindex_train, columns=[col_name])

    # 2) Frequenz ermitteln
    frequency = train_df.index[1] - train_df.index[0]
    minutes_per_step = frequency.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step) if minutes_per_step > 0 else 4

    # 3) Darts TimeSeries erstellen
    train_series = TimeSeries.from_dataframe(
        train_df.reset_index(),
        time_col="index",
        value_cols=col_name,
        fill_missing_dates=True,
        fillna_value=None,
    )

    # Interpolation für fehlende Werte
    df_interp = train_series.to_dataframe().interpolate("linear").bfill().ffill()
    train_series = TimeSeries.from_dataframe(df_interp)

    # 4) Skalierung
    scaler = Scaler()
    train_series_scaled = scaler.fit_transform(train_series)

    # 5) Encoder für Zeitfeatures (ähnlich Fourier-Encoding)
    encoders = {
        "cyclic": {
            "past": ["hour", "dayofweek", "month"],
            "future": ["hour", "dayofweek", "month"],
        },
        "datetime_attribute": {"past": ["dayofyear"], "future": ["dayofyear"]},
        "transformer": Scaler(),
    }

    # 6) Standard-Modellparameter (können überschrieben werden)
    default_params = {
        "lags": steps_per_hour * 24 * 2,  # 2 Tage Lags
        "output_chunk_length": horizon,
        "multi_models": True,
        "lags_past_covariates": [-24, -12, -6, -3, -1],
        "lags_future_covariates": list(range(0, horizon)),
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
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Benutzerdefinierte Parameter übernehmen
    if model_params:
        default_params.update(model_params)

    # 7) Modell erstellen und trainieren
    model = LightGBMModel(**default_params)
    model.fit(series=train_series_scaled)

    # 8) Prognose erstellen
    prediction_scaled = model.predict(n=horizon)

    # 9) Skalierung rückgängig machen
    prediction = scaler.inverse_transform(prediction_scaled)

    # 10) Als DataFrame mit korrektem Index zurückgeben
    prediction_df = prediction.to_dataframe()
    prediction_df.columns = [f"{col_name}_pred"]

    # Timezone konvertieren falls nötig
    if prediction_df.index.tz is None:
        prediction_df.index = prediction_df.index.tz_localize("UTC").tz_convert(
            "Europe/Berlin"
        )
    elif str(prediction_df.index.tz) != "Europe/Berlin":
        prediction_df.index = prediction_df.index.tz_convert("Europe/Berlin")

    return prediction_df


def get_unix_time_load(input_dict: Dict, fc_output: pd.DataFrame) -> Dict:
    """
    Konvertiert Prognose-Output in das gewünschte Format.

    Args:
        input_dict: Dictionary mit 'fc_data' Key
        fc_output: DataFrame von make_forecast()

    Returns:
        input_dict mit hinzugefügten Prognosedaten
    """
    # Spaltenname ermitteln (endet mit _pred)
    pred_col = [c for c in fc_output.columns if c.endswith("_pred")][0]

    input_dict["fc_data"]["load_fc_0"] = fc_output[pred_col].to_list()
    input_dict["fc_data"]["unixtime"] = (
        fc_output.index.astype("int64") // 10**9
    ).tolist()
    return input_dict


def run_hyperparameter_optimization(
    train_series: TimeSeries,
    horizon: int,
    steps_per_hour: int,
    n_trials: int = 30,
    timeout_seconds: int = 1800,
    tuning_objective: str = "RMSE",
    val_split: float = 0.8,
) -> Optional[Dict[str, Any]]:
    """
    Führt Hyperparameter-Optimierung mit Optuna durch.

    Args:
        train_series: Darts TimeSeries für Training
        horizon: Prognosehorizont in Schritten
        steps_per_hour: Datenpunkte pro Stunde
        n_trials: Anzahl Optuna-Trials
        timeout_seconds: Timeout in Sekunden
        tuning_objective: Optimierungsziel ('RMSE', 'MAPE', 'NRMSE')
        val_split: Train/Validation Split (0.8 = 80% Training)

    Returns:
        Dictionary mit besten Hyperparametern oder None
    """
    if not OPTUNA_AVAILABLE:
        log.warning("Optuna nicht installiert - HPO wird übersprungen.")
        return None

    log.info(
        f"Starte Optuna-Optimierung ({n_trials} Trials, Timeout: {timeout_seconds}s)"
    )

    # Train/Validation Split
    split_point = int(len(train_series) * val_split)
    train_hpo = train_series[:split_point]
    val_hpo = train_series[split_point:]

    log.info(
        f"HPO Training: {len(train_hpo)} Punkte, Validierung: {len(val_hpo)} Punkte"
    )

    # Minimale Validierungslänge sicherstellen
    min_val_length = 24 * steps_per_hour  # Min 1 Tag
    if len(val_hpo) < min_val_length:
        log.warning("Validierung zu kurz - Aufteilung wird angepasst.")
        split_point = len(train_series) - min_val_length
        train_hpo = train_series[:split_point]
        val_hpo = train_series[split_point:]

    steps_per_day = steps_per_hour * 24

    def objective(trial) -> float:
        # Lags: 1-2 Tage
        lags = trial.suggest_int("lags", steps_per_day * 1, steps_per_day * 2)

        output_chunk_length = horizon
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

        # Past Covariates Lags
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

            # Validierung mit Historical Forecasts
            forecasts = model.historical_forecasts(
                series=train_series,
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

            # Auf Validierungsbereich beschränken
            pred = pred.slice_intersect(val_hpo)

            # Metrik berechnen
            if tuning_objective == "RMSE":
                return float(rmse(val_hpo, pred))
            elif tuning_objective == "MAPE":
                return float(mape(val_hpo, pred))
            elif tuning_objective == "NRMSE":
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

    # Optuna Studie erstellen und ausführen
    study = optuna.create_study(
        direction="minimize",
        study_name="lgbm_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        n_jobs=1,
        show_progress_bar=True,
    )

    log.info(f"Beste Hyperparameter: {study.best_params}")
    log.info(f"Bester Score ({tuning_objective}): {study.best_value:.6f}")

    return study.best_params


def make_forecast_with_hpo(
    load: list,
    unixtime: list,
    horizon: int,
    col_name: str = "load",
    enable_hpo: bool = True,
    n_trials: int = 30,
    timeout_seconds: int = 1800,
    tuning_objective: str = "RMSE",
    val_split: float = 0.8,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Erstellt LightGBM-Prognose mit optionaler Hyperparameter-Optimierung.

    Args:
        load: Liste mit Lastwerten (kW)
        unixtime: Liste mit Unix-Timestamps (Sekunden)
        horizon: Anzahl der Prognoseschritte
        col_name: Spaltenname für die Zeitreihe
        enable_hpo: HPO aktivieren (True/False)
        n_trials: Anzahl Optuna-Trials
        timeout_seconds: Timeout für HPO in Sekunden
        tuning_objective: Optimierungsziel ('RMSE', 'MAPE', 'NRMSE')
        val_split: Train/Validation Split für HPO

    Returns:
        Tuple aus (Prognose-DataFrame, beste Hyperparameter oder None)
    """
    # 1) Trainingsdaten erstellen
    datetimeindex_train = pd.to_datetime(unixtime, unit="s", utc=True)
    datetimeindex_train = datetimeindex_train.tz_convert("Europe/Berlin")
    train_df = pd.DataFrame(data=load, index=datetimeindex_train, columns=[col_name])

    # 2) Frequenz ermitteln
    frequency = train_df.index[1] - train_df.index[0]
    minutes_per_step = frequency.total_seconds() / 60
    steps_per_hour = int(60 / minutes_per_step) if minutes_per_step > 0 else 4

    # 3) TimeSeries erstellen
    train_series = TimeSeries.from_dataframe(
        train_df.reset_index(),
        time_col="index",
        value_cols=col_name,
        fill_missing_dates=True,
        fillna_value=None,
    )
    df_interp = train_series.to_dataframe().interpolate("linear").bfill().ffill()
    train_series = TimeSeries.from_dataframe(df_interp)

    # 4) Skalierung
    scaler = Scaler()
    train_series_scaled = scaler.fit_transform(train_series)

    # 5) HPO durchführen (falls aktiviert)
    best_params = None
    if enable_hpo and OPTUNA_AVAILABLE:
        best_params = run_hyperparameter_optimization(
            train_series=train_series_scaled,
            horizon=horizon,
            steps_per_hour=steps_per_hour,
            n_trials=n_trials,
            timeout_seconds=timeout_seconds,
            tuning_objective=tuning_objective,
            val_split=val_split,
        )

    # 6) Encoder
    encoders = {
        "cyclic": {
            "past": ["hour", "dayofweek", "month"],
            "future": ["hour", "dayofweek", "month"],
        },
        "datetime_attribute": {"past": ["dayofyear"], "future": ["dayofyear"]},
        "transformer": Scaler(),
    }

    # 7) Modellparameter aus HPO oder Defaults
    if best_params:
        # Past-Lags aus HPO rekonstruieren
        past_lag_depth = int(best_params.get("past_lag_depth", 3))
        past_cov_lags = []
        for i in range(past_lag_depth):
            lag = best_params.get(f"past_lag_{i}")
            if lag is not None:
                past_cov_lags.append(int(lag))
        past_cov_lags = (
            sorted(past_cov_lags) if past_cov_lags else [-24, -12, -6, -3, -1]
        )

        future_cov_lags = list(range(0, horizon))

        model_params = {
            "lags": int(best_params.get("lags", steps_per_hour * 24)),
            "output_chunk_length": horizon,
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
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    else:
        # Fallback: Standard-Parameter
        model_params = {
            "lags": steps_per_hour * 24 * 2,
            "output_chunk_length": horizon,
            "multi_models": True,
            "lags_past_covariates": [-24, -12, -6, -3, -1],
            "lags_future_covariates": list(range(0, horizon)),
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
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    # 8) Modell trainieren
    model = LightGBMModel(**model_params)
    model.fit(series=train_series_scaled)

    # 9) Prognose
    prediction_scaled = model.predict(n=horizon)
    prediction = scaler.inverse_transform(prediction_scaled)

    # 10) Ergebnis formatieren
    prediction_df = prediction.to_dataframe()
    prediction_df.columns = [f"{col_name}_pred"]

    if prediction_df.index.tz is None:
        prediction_df.index = prediction_df.index.tz_localize("UTC").tz_convert(
            "Europe/Berlin"
        )
    elif str(prediction_df.index.tz) != "Europe/Berlin":
        prediction_df.index = prediction_df.index.tz_convert("Europe/Berlin")

    return prediction_df, best_params


def make_forecast_with_holidays(
    load: list,
    unixtime: list,
    horizon: int,
    col_name: str = "load",
    state: str = "BW",
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Erweiterte Version mit Feiertags-Features (benötigt 'holidays' Package).

    Args:
        load: Liste mit Lastwerten (kW)
        unixtime: Liste mit Unix-Timestamps (Sekunden)
        horizon: Anzahl der Prognoseschritte
        col_name: Spaltenname für die Zeitreihe
        state: Bundesland für Feiertage (z.B. 'BW' für Baden-Württemberg)
        model_params: Optionale LightGBM-Parameter

    Returns:
        DataFrame mit Prognose
    """
    try:
        import holidays

        # Trainingsdaten erstellen
        datetimeindex_train = pd.to_datetime(unixtime, unit="s", utc=True)
        datetimeindex_train = datetimeindex_train.tz_convert("Europe/Berlin")
        train_df = pd.DataFrame(
            data=load, index=datetimeindex_train, columns=[col_name]
        )

        # Frequenz ermitteln
        frequency = train_df.index[1] - train_df.index[0]
        minutes_per_step = frequency.total_seconds() / 60
        steps_per_hour = int(60 / minutes_per_step) if minutes_per_step > 0 else 4

        # Feiertage für relevante Jahre
        years = list(set(train_df.index.year.tolist()))
        # Auch Folgejahr für Prognose berücksichtigen
        years.extend([max(years) + 1])
        de_holidays = holidays.Germany(state=state, years=years)

        # Holiday-Feature hinzufügen
        train_df["is_holiday"] = (
            train_df.index.date.astype("datetime64[ns]").isin(de_holidays).astype(int)
        )

        # TimeSeries erstellen
        train_series = TimeSeries.from_dataframe(
            train_df.reset_index(),
            time_col="index",
            value_cols=col_name,
            fill_missing_dates=True,
            fillna_value=None,
        )

        # Holiday-Covariate separat
        holiday_series = TimeSeries.from_dataframe(
            train_df[["is_holiday"]].reset_index(),
            time_col="index",
            value_cols="is_holiday",
            fill_missing_dates=True,
            fillna_value=0,
        )

        # Interpolation
        df_interp = train_series.to_dataframe().interpolate("linear").bfill().ffill()
        train_series = TimeSeries.from_dataframe(df_interp)

        # Skalierung
        scaler = Scaler()
        train_series_scaled = scaler.fit_transform(train_series)

        # Encoder
        encoders = {
            "cyclic": {
                "past": ["hour", "dayofweek", "month"],
                "future": ["hour", "dayofweek", "month"],
            },
            "datetime_attribute": {"past": ["dayofyear"], "future": ["dayofyear"]},
            "transformer": Scaler(),
        }

        # Modellparameter
        default_params = {
            "lags": steps_per_hour * 24 * 2,
            "output_chunk_length": horizon,
            "multi_models": True,
            "lags_past_covariates": [-24, -12, -6, -3, -1],
            "lags_future_covariates": list(range(0, horizon)),
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
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        if model_params:
            default_params.update(model_params)

        # Modell trainieren
        model = LightGBMModel(**default_params)
        model.fit(series=train_series_scaled, past_covariates=holiday_series)

        # Future holidays für Prognose generieren
        first_fc_index = train_df.index[-1] + frequency
        last_fc_index = first_fc_index + (horizon - 1) * frequency
        fc_index = pd.date_range(
            start=first_fc_index, end=last_fc_index, freq=frequency
        )

        future_holidays_df = pd.DataFrame(index=fc_index)
        future_holidays_df["is_holiday"] = (
            future_holidays_df.index.date.astype("datetime64[ns]")
            .isin(de_holidays)
            .astype(int)
        )

        # Kombinierte Holiday-Serie (Training + Prognose)
        combined_holiday_df = pd.concat([train_df[["is_holiday"]], future_holidays_df])
        combined_holiday_series = TimeSeries.from_dataframe(
            combined_holiday_df.reset_index(), time_col="index", value_cols="is_holiday"
        )

        # Prognose
        prediction_scaled = model.predict(
            n=horizon, past_covariates=combined_holiday_series
        )
        prediction = scaler.inverse_transform(prediction_scaled)

        # Ergebnis formatieren
        prediction_df = prediction.to_dataframe()
        prediction_df.columns = [f"{col_name}_pred"]

        if prediction_df.index.tz is None:
            prediction_df.index = prediction_df.index.tz_localize("UTC").tz_convert(
                "Europe/Berlin"
            )
        elif str(prediction_df.index.tz) != "Europe/Berlin":
            prediction_df.index = prediction_df.index.tz_convert("Europe/Berlin")

        return prediction_df

    except ImportError:
        # Fallback ohne Feiertage
        return make_forecast(load, unixtime, horizon, col_name, model_params)


# Beispielnutzung
if __name__ == "__main__":
    # Testdaten generieren (simuliert)
    import numpy as np

    # 1 Woche Testdaten (15-Minuten-Intervalle)
    n_points = 7 * 24 * 4  # 672 Punkte
    start_time = pd.Timestamp("2024-01-01", tz="Europe/Berlin")

    # Unix-Timestamps
    timestamps = pd.date_range(start=start_time, periods=n_points, freq="15min")
    unixtime = (timestamps.astype("int64") // 10**9).tolist()

    # Simulierte Last mit Tages- und Wochenmuster
    t = np.arange(n_points)
    daily_pattern = 2 * np.sin(2 * np.pi * t / (24 * 4))  # Tageszyklus
    weekly_pattern = 0.5 * np.sin(2 * np.pi * t / (7 * 24 * 4))  # Wochenzyklus
    noise = 0.2 * np.random.randn(n_points)
    load = (5 + daily_pattern + weekly_pattern + noise).tolist()

    # Prognose erstellen (24 Stunden = 96 Schritte bei 15-Min-Auflösung)
    horizon = 24 * 4

    # Option 1: Einfache Prognose ohne HPO
    print("=" * 60)
    print("Option 1: Einfache Prognose ohne HPO")
    print("=" * 60)
    prediction = make_forecast(load, unixtime, horizon)
    print(f"Prognose erstellt: {len(prediction)} Datenpunkte")
    print(f"Von: {prediction.index[0]}")
    print(f"Bis: {prediction.index[-1]}")
    print(f"\nErste 5 Prognosewerte:\n{prediction.head()}")

    # Option 2: Prognose mit HPO (nur ausführen wenn gewünscht)
    RUN_HPO_EXAMPLE = False  # Auf True setzen für HPO-Test
    if RUN_HPO_EXAMPLE and OPTUNA_AVAILABLE:
        print("\n" + "=" * 60)
        print("Option 2: Prognose mit Hyperparameter-Optimierung")
        print("=" * 60)
        prediction_hpo, best_params = make_forecast_with_hpo(
            load,
            unixtime,
            horizon,
            enable_hpo=True,
            n_trials=10,  # Weniger Trials für Demo
            timeout_seconds=300,
            tuning_objective="RMSE",
        )
        print(f"\nPrognose mit HPO erstellt: {len(prediction_hpo)} Datenpunkte")
        print(f"Beste Parameter: {best_params}")
