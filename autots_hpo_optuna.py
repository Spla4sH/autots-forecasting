# AutoTS mit Hyperparameter-Optimierung (HPO) via Optuna
# Optimiert AutoTS-Parameter für beste Vorhersagequalität

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autots import AutoTS
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings

warnings.filterwarnings("ignore")


# Beispieldaten erstellen
def create_sample_data():
    """Erstellt Beispiel-Stromlastdaten für einen typischen Haushalt"""
    date_range = pd.date_range(start="2023-10-01", end="2023-12-31 23:00", freq="h")

    base_load = 0.5
    hours = np.array([dt.hour for dt in date_range])
    minutes = np.array([dt.minute for dt in date_range])
    hour_of_day = hours + minutes / 60

    morning_peak = 2.0 * np.exp(-((hour_of_day - 7.5) ** 2) / 2)
    evening_peak = 3.5 * np.exp(-((hour_of_day - 20) ** 2) / 3)
    night_reduction = -0.3 * ((hour_of_day >= 23) | (hour_of_day <= 6)).astype(float)

    weekdays = np.array([dt.dayofweek for dt in date_range])
    is_weekend = (weekdays >= 5).astype(float)
    weekend_pattern = is_weekend * 0.8

    day_of_year = np.array([dt.dayofyear for dt in date_range])
    winter_heating = 1.5 * np.exp(-((day_of_year - 15) ** 2) / 5000) + 1.5 * np.exp(
        -((day_of_year - 365) ** 2) / 5000
    )
    summer_cooling = 1.0 * np.exp(-((day_of_year - 200) ** 2) / 3000)

    random_appliances = np.random.choice(
        [0, 0, 0, 0, 1.5, 2.0, 2.5],
        size=len(date_range),
        p=[0.85, 0.05, 0.04, 0.03, 0.015, 0.01, 0.005],
    )

    noise = np.random.normal(0, 0.15, len(date_range))

    power_load = (
        base_load
        + morning_peak
        + evening_peak
        + night_reduction
        + weekend_pattern
        + winter_heating
        + summer_cooling
        + random_appliances
        + noise
    )

    power_load = np.maximum(power_load, 0.1)

    df = pd.DataFrame({"datetime": date_range, "power_load_kw": power_load})
    df = df.set_index("datetime")

    return df


def objective(trial, train_data, val_data):
    """
    Optuna Objective Function für AutoTS Hyperparameter-Optimierung

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna Trial Objekt
    train_data : DataFrame
        Trainingsdaten
    val_data : DataFrame
        Validierungsdaten
    """

    # Hyperparameter vorschlagen
    forecast_length = 24  # Fest: 24 Stunden

    # Model List Auswahl
    model_list = trial.suggest_categorical(
        "model_list", ["superfast", "fast", "default"]
    )

    # Ensemble Typ
    ensemble = trial.suggest_categorical(
        "ensemble", ["simple", "distance", "horizontal"]
    )

    # Generationen
    max_generations = trial.suggest_int("max_generations", 2, 5)

    # Validierungen
    num_validations = trial.suggest_int("num_validations", 1, 3)

    # Prediction Interval
    prediction_interval = trial.suggest_float("prediction_interval", 0.90, 0.99)

    # Transformer List
    transformer_list = trial.suggest_categorical(
        "transformer_list", ["fast", "all", "superfast"]
    )

    # Drop most recent (für Robustheit)
    drop_most_recent = trial.suggest_int("drop_most_recent", 0, 2)

    try:
        # Modell mit vorgeschlagenen Parametern erstellen
        model = AutoTS(
            forecast_length=forecast_length,
            frequency="infer",
            prediction_interval=prediction_interval,
            ensemble=ensemble,
            max_generations=max_generations,
            num_validations=num_validations,
            validation_method="backwards",
            model_list=model_list,
            transformer_list=transformer_list,
            drop_most_recent=drop_most_recent,
            n_jobs=1,
            verbose=0,  # Kein Output während HPO
        )

        # Training
        model = model.fit(train_data)

        # Vorhersage auf Validierungsdaten
        prediction = model.predict()

        # Validierungsfehler berechnen
        # Nehme nur die ersten 24 Stunden der Validierungsdaten
        actual = val_data.iloc[:forecast_length]["power_load_kw"].values
        predicted = prediction.forecast["power_load_kw"].values[:forecast_length]

        # Metriken berechnen
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))

        # Kombinierte Metrik (gewichtet)
        combined_metric = 0.6 * mae + 0.4 * rmse

        # Speichere zusätzliche Metriken für Analyse
        trial.set_user_attr("mae", mae)
        trial.set_user_attr("rmse", rmse)
        trial.set_user_attr("best_model", model.best_model_name)

        return combined_metric

    except Exception as e:
        # Bei Fehler: hohe Strafe
        print(f"Trial {trial.number} failed: {e}")
        return float("inf")


def run_hyperparameter_optimization(df, n_trials=20):
    """
    Führt Hyperparameter-Optimierung mit Optuna durch

    Parameters:
    -----------
    df : DataFrame
        Kompletter Datensatz
    n_trials : int
        Anzahl der Optuna Trials
    """

    print("=== HYPERPARAMETER-OPTIMIERUNG MIT OPTUNA ===\n")

    # Split in Train/Val (80/20)
    split_idx = int(len(df) * 0.8)
    train_data = df.iloc[:split_idx]
    val_data = df.iloc[split_idx:]

    print(
        f"Trainingsdaten: {len(train_data)} Stunden ({train_data.index[0]} bis {train_data.index[-1]})"
    )
    print(
        f"Validierungsdaten: {len(val_data)} Stunden ({val_data.index[0]} bis {val_data.index[-1]})"
    )
    print(f"\nOptuna Trials: {n_trials}\n")

    # Optuna Study erstellen
    study = optuna.create_study(
        direction="minimize",  # Minimiere kombinierten Fehler
        study_name="autots_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    )

    # Optimierung durchführen
    study.optimize(
        lambda trial: objective(trial, train_data, val_data),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Beste Parameter
    print("\n=== OPTIMIERUNGSERGEBNISSE ===\n")
    print("Beste Parameter:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nBester kombinierter Score: {study.best_trial.value:.4f}")
    print(f"  MAE: {study.best_trial.user_attrs['mae']:.4f} kW")
    print(f"  RMSE: {study.best_trial.user_attrs['rmse']:.4f} kW")
    print(f"  Bestes Modell: {study.best_trial.user_attrs['best_model']}")

    return study, train_data, val_data


def train_final_model(study, train_data, val_data):
    """
    Trainiert finales Modell mit besten Parametern
    """

    print("\n=== TRAINING FINALES MODELL ===\n")

    best_params = study.best_params

    # Finales Modell mit besten Parametern
    final_model = AutoTS(
        forecast_length=24,
        frequency="infer",
        prediction_interval=best_params["prediction_interval"],
        ensemble=best_params["ensemble"],
        max_generations=best_params["max_generations"],
        num_validations=best_params["num_validations"],
        validation_method="backwards",
        model_list=best_params["model_list"],
        transformer_list=best_params["transformer_list"],
        drop_most_recent=best_params["drop_most_recent"],
        n_jobs=1,
        verbose=1,
    )

    # Training auf kompletten Trainingsdaten
    print("Training mit optimierten Parametern...")
    final_model = final_model.fit(train_data)

    # Test auf Validierungsdaten
    prediction = final_model.predict()

    actual = val_data.iloc[:24]["power_load_kw"].values
    predicted = prediction.forecast["power_load_kw"].values[:24]

    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    print(f"\nFinale Test-Metriken:")
    print(f"  MAE:  {mae:.4f} kW")
    print(f"  RMSE: {rmse:.4f} kW")
    print(f"  MAPE: {mape:.2f}%")

    return final_model, prediction


def visualize_optimization(study):
    """
    Erstellt Visualisierungen der Optimierung
    """

    print("\n=== ERSTELLE VISUALISIERUNGEN ===")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Optimization History
    ax1 = axes[0, 0]
    trials = study.trials
    values = [t.value for t in trials if t.value != float("inf")]
    ax1.plot(range(1, len(values) + 1), values, marker="o", linewidth=2, markersize=6)
    ax1.axhline(
        y=study.best_value,
        color="r",
        linestyle="--",
        label=f"Best: {study.best_value:.4f}",
    )
    ax1.set_xlabel("Trial Nummer")
    ax1.set_ylabel("Kombinierter Score (MAE*0.6 + RMSE*0.4)")
    ax1.set_title("Optuna Optimization History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter Importance
    ax2 = axes[0, 1]
    importances = optuna.importance.get_param_importances(study)
    params = list(importances.keys())
    importance_values = list(importances.values())

    ax2.barh(params, importance_values, color="steelblue")
    ax2.set_xlabel("Importance")
    ax2.set_title("Hyperparameter Importance")
    ax2.grid(True, alpha=0.3, axis="x")

    # Plot 3: MAE Distribution
    ax3 = axes[1, 0]
    mae_values = [
        t.user_attrs.get("mae", np.nan) for t in trials if "mae" in t.user_attrs
    ]
    ax3.hist(mae_values, bins=20, color="green", alpha=0.7, edgecolor="black")
    ax3.axvline(
        x=study.best_trial.user_attrs["mae"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best: {study.best_trial.user_attrs['mae']:.4f}",
    )
    ax3.set_xlabel("MAE (kW)")
    ax3.set_ylabel("Anzahl Trials")
    ax3.set_title("MAE Verteilung über alle Trials")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: RMSE Distribution
    ax4 = axes[1, 1]
    rmse_values = [
        t.user_attrs.get("rmse", np.nan) for t in trials if "rmse" in t.user_attrs
    ]
    ax4.hist(rmse_values, bins=20, color="purple", alpha=0.7, edgecolor="black")
    ax4.axvline(
        x=study.best_trial.user_attrs["rmse"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best: {study.best_trial.user_attrs['rmse']:.4f}",
    )
    ax4.set_xlabel("RMSE (kW)")
    ax4.set_ylabel("Anzahl Trials")
    ax4.set_title("RMSE Verteilung über alle Trials")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optuna_optimization.png", dpi=300, bbox_inches="tight")
    print("  Gespeichert als 'optuna_optimization.png'")


# Hauptprogramm
if __name__ == "__main__":
    print("AutoTS mit Optuna Hyperparameter-Optimierung\n")

    # 1. Daten erstellen
    print("1. Erstelle Datensatz...")
    df = create_sample_data()
    print(f"   Datensatz: {len(df)} Stunden\n")

    # 2. Hyperparameter-Optimierung
    study, train_data, val_data = run_hyperparameter_optimization(
        df, n_trials=15  # Mehr Trials = bessere Optimierung, aber langsamer
    )

    # 3. Finales Modell trainieren
    final_model, prediction = train_final_model(study, train_data, val_data)

    # 4. Visualisierungen
    visualize_optimization(study)

    # 5. Study speichern
    import joblib

    joblib.dump(study, "optuna_study.pkl")
    print("\n  Optuna Study gespeichert als 'optuna_study.pkl'")

    # 6. Beste Parameter als JSON speichern
    import json

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("  Beste Parameter gespeichert als 'best_params.json'")

    print("\n✓ HPO abgeschlossen!")
    print("\n=== VERWENDUNG DER BESTEN PARAMETER ===")
    print("Die besten Parameter können nun in autots_example.py verwendet werden:")
    print(f"\nmodel = AutoTS(")
    for key, value in study.best_params.items():
        if isinstance(value, str):
            print(f"    {key}='{value}',")
        else:
            print(f"    {key}={value},")
    print("    ...)")
