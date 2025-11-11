# AutoTS mit Rolling Window und wöchentlichem Retraining
# Simuliert produktives Szenario mit regelmäßigem Model-Update

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autots import AutoTS
import time


# Beispieldaten erstellen
def create_sample_data():
    """Erstellt Beispiel-Stromlastdaten für einen typischen Haushalt"""
    # Stündliche Daten über 6 Monate für Rolling Window
    date_range = pd.date_range(start="2023-07-01", end="2023-12-31 23:00", freq="h")

    # Basis-Last eines Haushalts (kW)
    base_load = 0.5

    # Tageszeit-abhängiges Muster
    hours = np.array([dt.hour for dt in date_range])
    minutes = np.array([dt.minute for dt in date_range])
    hour_of_day = hours + minutes / 60

    # Morgens (6-9 Uhr): Erhöhter Verbrauch
    morning_peak = 2.0 * np.exp(-((hour_of_day - 7.5) ** 2) / 2)

    # Abends (18-22 Uhr): Hauptverbrauch
    evening_peak = 3.5 * np.exp(-((hour_of_day - 20) ** 2) / 3)

    # Nachtabsenkung
    night_reduction = -0.3 * ((hour_of_day >= 23) | (hour_of_day <= 6)).astype(float)

    # Wochentag vs. Wochenende
    weekdays = np.array([dt.dayofweek for dt in date_range])
    is_weekend = (weekdays >= 5).astype(float)
    weekend_pattern = is_weekend * 0.8

    # Saisonale Schwankungen (Winter/Sommer)
    day_of_year = np.array([dt.dayofyear for dt in date_range])
    winter_heating = 1.5 * np.exp(-((day_of_year - 15) ** 2) / 5000) + 1.5 * np.exp(
        -((day_of_year - 365) ** 2) / 5000
    )
    summer_cooling = 1.0 * np.exp(-((day_of_year - 200) ** 2) / 3000)

    # Zufällige Spitzen
    random_appliances = np.random.choice(
        [0, 0, 0, 0, 1.5, 2.0, 2.5],
        size=len(date_range),
        p=[0.85, 0.05, 0.04, 0.03, 0.015, 0.01, 0.005],
    )

    # Zufälliges Rauschen
    noise = np.random.normal(0, 0.15, len(date_range))

    # Kombiniere alle Komponenten
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


def rolling_window_forecast(df, initial_train_weeks=8, forecast_horizon=24, retrain_weeks=1):
    """
    Führt Rolling Window Forecasting mit regelmäßigem Retraining durch
    
    Parameters:
    -----------
    df : DataFrame
        Kompletter Datensatz
    initial_train_weeks : int
        Anzahl Wochen für initiales Training
    forecast_horizon : int
        Vorhersagehorizont in Stunden
    retrain_weeks : int
        Retraining alle X Wochen
    """
    
    initial_train_hours = initial_train_weeks * 7 * 24
    retrain_hours = retrain_weeks * 7 * 24
    
    # Initialer Trainingszeitraum
    train_end_idx = initial_train_hours
    
    predictions = []
    actuals = []
    retrain_points = []
    metrics_history = []
    
    iteration = 0
    total_iterations = (len(df) - initial_train_hours) // retrain_hours
    
    print(f"\n=== ROLLING WINDOW FORECAST ===")
    print(f"Initiales Training: {initial_train_weeks} Wochen ({initial_train_hours} Stunden)")
    print(f"Retraining alle: {retrain_weeks} Woche(n) ({retrain_hours} Stunden)")
    print(f"Forecast Horizon: {forecast_horizon} Stunden")
    print(f"Geplante Iterationen: {total_iterations}\n")
    
    while train_end_idx + forecast_horizon <= len(df):
        iteration += 1
        
        # Trainingsdaten (Rolling Window)
        train_data = df.iloc[:train_end_idx]
        
        print(f"--- Iteration {iteration}/{total_iterations} ---")
        print(f"Training: {train_data.index[0]} bis {train_data.index[-1]}")
        print(f"Forecast für: {df.index[train_end_idx]} bis {df.index[train_end_idx + forecast_horizon - 1]}")
        
        # Modell trainieren
        start_time = time.time()
        model = AutoTS(
            forecast_length=forecast_horizon,
            frequency="infer",
            prediction_interval=0.95,
            ensemble="simple",
            max_generations=2,  # Schneller für Rolling Window
            num_validations=1,
            validation_method="backwards",
            model_list="superfast",
            transformer_list="fast",
            drop_most_recent=0,
            n_jobs=1,
            verbose=0,  # Weniger Output
        )
        
        model = model.fit(train_data)
        train_time = time.time() - start_time
        
        # Vorhersage
        prediction = model.predict()
        
        # Echte Werte für Vergleich
        actual = df.iloc[train_end_idx:train_end_idx + forecast_horizon]
        
        # Metriken berechnen
        mae = np.mean(np.abs(prediction.forecast['power_load_kw'].values - actual['power_load_kw'].values))
        rmse = np.sqrt(np.mean((prediction.forecast['power_load_kw'].values - actual['power_load_kw'].values) ** 2))
        mape = np.mean(np.abs((actual['power_load_kw'].values - prediction.forecast['power_load_kw'].values) / actual['power_load_kw'].values)) * 100
        
        print(f"Bestes Modell: {model.best_model_name}")
        print(f"Training Zeit: {train_time:.1f}s")
        print(f"MAE: {mae:.3f} kW, RMSE: {rmse:.3f} kW, MAPE: {mape:.2f}%")
        print()
        
        # Ergebnisse speichern
        predictions.append(prediction.forecast['power_load_kw'])
        actuals.append(actual['power_load_kw'])
        retrain_points.append(df.index[train_end_idx])
        metrics_history.append({
            'iteration': iteration,
            'train_end': train_data.index[-1],
            'forecast_start': df.index[train_end_idx],
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_time': train_time,
            'model': model.best_model_name
        })
        
        # Nächstes Fenster (um retrain_hours verschieben)
        train_end_idx += retrain_hours
    
    return predictions, actuals, retrain_points, metrics_history


# Hauptprogramm
if __name__ == "__main__":
    print("AutoTS Rolling Window mit wöchentlichem Retraining\n")
    
    # 1. Daten erstellen
    print("1. Erstelle Beispiel-Stromlastdaten...")
    df = create_sample_data()
    print(f"   Datensatz: {len(df)} Stunden von {df.index[0]} bis {df.index[-1]}")
    print(f"   Durchschnittliche Last: {df['power_load_kw'].mean():.3f} kW\n")
    
    # 2. Rolling Window Forecast
    predictions, actuals, retrain_points, metrics_history = rolling_window_forecast(
        df,
        initial_train_weeks=8,   # 8 Wochen initiales Training
        forecast_horizon=24,      # 24 Stunden Vorhersage
        retrain_weeks=1           # Jede Woche neu trainieren
    )
    
    # 3. Gesamtmetriken
    print("\n=== GESAMTERGEBNISSE ===")
    all_predictions = pd.concat(predictions)
    all_actuals = pd.concat(actuals)
    
    overall_mae = np.mean(np.abs(all_predictions.values - all_actuals.values))
    overall_rmse = np.sqrt(np.mean((all_predictions.values - all_actuals.values) ** 2))
    overall_mape = np.mean(np.abs((all_actuals.values - all_predictions.values) / all_actuals.values)) * 100
    
    print(f"Anzahl Retrainings: {len(metrics_history)}")
    print(f"Gesamte vorhergesagte Stunden: {len(all_predictions)}")
    print(f"\nDurchschnittliche Metriken:")
    print(f"  MAE:  {overall_mae:.3f} kW")
    print(f"  RMSE: {overall_rmse:.3f} kW")
    print(f"  MAPE: {overall_mape:.2f}%")
    
    avg_train_time = np.mean([m['train_time'] for m in metrics_history])
    print(f"\nDurchschnittliche Trainingszeit: {avg_train_time:.1f}s pro Woche")
    
    # 4. Metriken über Zeit
    metrics_df = pd.DataFrame(metrics_history)
    print(f"\nMetriken-Trend:")
    print(f"  Beste MAE:  {metrics_df['mae'].min():.3f} kW (Iteration {metrics_df.loc[metrics_df['mae'].idxmin(), 'iteration']})")
    print(f"  Schlechteste MAE: {metrics_df['mae'].max():.3f} kW (Iteration {metrics_df.loc[metrics_df['mae'].idxmax(), 'iteration']})")
    
    # 5. Visualisierung
    print("\n4. Erstelle Visualisierungen...")
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Vorhersage vs. Tatsächliche Werte (letzte 2 Wochen)
    ax1 = axes[0]
    last_two_weeks_hours = 14 * 24
    plot_start = max(0, len(all_actuals) - last_two_weeks_hours)
    
    ax1.plot(all_actuals.index[plot_start:], all_actuals.values[plot_start:], 
             label='Tatsächliche Werte', color='blue', linewidth=1.5, alpha=0.8)
    ax1.plot(all_predictions.index[plot_start:], all_predictions.values[plot_start:], 
             label='Vorhersage', color='red', linewidth=1.5, linestyle='--', alpha=0.8)
    
    # Markiere Retraining-Punkte
    for rp in retrain_points:
        if rp in all_actuals.index[plot_start:]:
            ax1.axvline(x=rp, color='green', alpha=0.3, linestyle=':', linewidth=1)
    
    ax1.set_xlabel('Datum und Uhrzeit')
    ax1.set_ylabel('Stromlast (kW)')
    ax1.set_title('Rolling Window Forecast: Vorhersage vs. Realität (letzte 2 Wochen)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Metriken über Zeit
    ax2 = axes[1]
    ax2.plot(range(1, len(metrics_history) + 1), metrics_df['mae'], 
             marker='o', label='MAE', color='blue', linewidth=2)
    ax2.plot(range(1, len(metrics_history) + 1), metrics_df['rmse'], 
             marker='s', label='RMSE', color='red', linewidth=2)
    
    ax2.set_xlabel('Retraining Iteration (wöchentlich)')
    ax2.set_ylabel('Fehler (kW)')
    ax2.set_title('Vorhersagefehler über Zeit (MAE & RMSE)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fehlerverteilung
    ax3 = axes[2]
    errors = all_predictions.values - all_actuals.values
    ax3.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Null-Fehler')
    ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Durchschnitt: {np.mean(errors):.3f} kW')
    
    ax3.set_xlabel('Vorhersagefehler (kW)')
    ax3.set_ylabel('Häufigkeit')
    ax3.set_title('Verteilung der Vorhersagefehler')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_window_forecast.png', dpi=300, bbox_inches='tight')
    print("   Grafiken gespeichert als 'rolling_window_forecast.png'")
    
    # 6. Metriken-Tabelle speichern
    metrics_df.to_csv('rolling_window_metrics.csv', index=False)
    print("   Metriken gespeichert als 'rolling_window_metrics.csv'")
    
    print("\n✓ Fertig!")
