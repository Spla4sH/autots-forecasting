# AutoTS Beispiel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autots import AutoTS
from scipy.stats import pearsonr


# Beispieldaten erstellen
def create_sample_data():
    """Erstellt Beispiel-Stromlastdaten für einen typischen Haushalt"""
    # Stündliche Daten über 3 Monate zum testen
    date_range = pd.date_range(start="2023-10-01", end="2023-12-31 23:00", freq="h")

    # Basislast eines Haushalts (kW)
    base_load = 0.5  # Standby-Verbrauch (Kühlschrank, Router, etc.)

    # Tageszeitabhängiges Muster
    hours = np.array([dt.hour for dt in date_range])
    minutes = np.array([dt.minute for dt in date_range])
    hour_of_day = hours + minutes / 60

    # Morgens (6-9 Uhr): Erhöhter Verbrauch (Frühstück, Duschen)
    morning_peak = 2.0 * np.exp(-((hour_of_day - 7.5) ** 2) / 2)

    # Abends (18-22 Uhr): Hauptverbrauch (Kochen, TV, Beleuchtung)
    evening_peak = 3.5 * np.exp(-((hour_of_day - 20) ** 2) / 3)

    # Nachtabsenkung (23-6 Uhr): Minimaler Verbrauch
    night_reduction = -0.3 * ((hour_of_day >= 23) | (hour_of_day <= 6)).astype(float)

    # Wochentag vs. Wochenende
    weekdays = np.array([dt.dayofweek for dt in date_range])
    is_weekend = (weekdays >= 5).astype(float)
    weekend_pattern = is_weekend * 0.8  # Am Wochenende mehr tagsüber zu Hause

    # Saisonale Schwankungen (Heizung im Winter, Klimaanlage im Sommer)
    day_of_year = np.array([dt.dayofyear for dt in date_range])
    # Winter (höherer Verbrauch durch Heizung)
    winter_heating = 1.5 * np.exp(-((day_of_year - 15) ** 2) / 5000) + 1.5 * np.exp(
        -((day_of_year - 365) ** 2) / 5000
    )
    # Sommer (höherer Verbrauch durch Klimaanlage)
    summer_cooling = 1.0 * np.exp(-((day_of_year - 200) ** 2) / 3000)

    # Zufällige Spitzen (Waschmaschine, Trockner, Geschirrspüler, etc.)
    random_appliances = np.random.choice(
        [0, 0, 0, 0, 1.5, 2.0, 2.5],  # Meistens 0, manchmal Großgeräte
        size=len(date_range),
        p=[0.85, 0.05, 0.04, 0.03, 0.015, 0.01, 0.005],
    )

    # Zufälliges Rauschen (kleine Schwankungen)
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

    # Stelle sicher, dass keine negativen Werte entstehen
    power_load = np.maximum(power_load, 0.1)

    df = pd.DataFrame({"datetime": date_range, "power_load_kw": power_load})
    df = df.set_index("datetime")

    return df


# Hauptprogramm
if __name__ == "__main__":
    print("AutoTS Haushalts-Stromlast-Vorhersage gestartet...\n")

    # 1. Daten erstellen
    print("1. Erstelle Beispiel-Stromlastdaten für Haushalt...")
    df = create_sample_data()
    print(f"   Datensatz: {len(df)} Messwerte (stündlich)")
    print(f"   Zeitraum: {df.index[0]} bis {df.index[-1]}")
    print(f"   Durchschnittliche Last: {df['power_load_kw'].mean():.3f} kW")
    print(
        f"   Min/Max Last: {df['power_load_kw'].min():.3f} / {df['power_load_kw'].max():.3f} kW"
    )
    print(f"   Tagesverbrauch (Ø): {df['power_load_kw'].mean() * 24:.2f} kWh")
    print(f"\n   Erste Zeilen:\n{df.head()}\n")

    # 2. AutoTS Modell konfigurieren
    print("2. Konfiguriere AutoTS Modell...")
    model = AutoTS(
        forecast_length=24,  # 1 Tag voraus (24 Stunden)
        frequency="infer",  # Frequenz automatisch erkennen
        prediction_interval=0.95,  # 95% Konfidenzintervall
        ensemble="simple",  # Einfaches Ensemble
        max_generations=3,  # Weniger Generationen für schnelleres Training
        num_validations=2,  # Validierungsrunden
        validation_method="backwards",  # Rückwärts-Validierung
        model_list="superfast",  # Sehr schnelle Modelle
        transformer_list="fast",  # Schnelle Transformationen
        drop_most_recent=0,  # Keine aktuellen Daten entfernen
        n_jobs=1,  # Keine Parallelisierung (vermeidet Windows-Probleme)
        verbose=1,  # Fortschrittsausgabe
    )

    # 3. Modell trainieren
    print("\n3. Trainiere Modell (kann einige Minuten dauern)...")
    model = model.fit(df)

    # 4. Vorhersage erstellen
    print("\n4. Erstelle Vorhersage...")
    prediction = model.predict()

    # 5. Ergebnisse anzeigen
    print("\n5. Ergebnisse:")
    print("\n   Beste Modelle:")
    print(model.best_model)

    # Modell-Metriken anzeigen
    print("\n   === MODELL-METRIKEN ===")
    print(f"   Bestes Modell: {model.best_model_name}")
    print(f"   Ensemble-Typ: {model.best_model['Ensemble'].values[0]}")

    # Validierungsmetriken
    validation_results = model.results()

    # Berechne zusätzliche Metriken auf den Validierungsdaten
    # Hole die letzten 24 Stunden als Testdaten für Metrik-Berechnung
    test_data = df.iloc[-24:]
    test_actual = test_data["power_load_kw"].values

    # Erstelle eine Vorhersage für die Testperiode (Back-Testing)
    train_data = df.iloc[:-24]
    temp_model = AutoTS(
        forecast_length=24,
        frequency="infer",
        prediction_interval=0.95,
        ensemble="simple",
        max_generations=3,
        num_validations=2,
        validation_method="backwards",
        model_list="superfast",
        transformer_list="fast",
        drop_most_recent=0,
        n_jobs=1,
        verbose=0,
    )
    temp_model = temp_model.fit(train_data)
    temp_prediction = temp_model.predict()
    test_forecast = temp_prediction.forecast["power_load_kw"].values

    # Berechne Metriken
    rmse = np.sqrt(np.mean((test_actual - test_forecast) ** 2))
    nrmse = rmse / (test_actual.max() - test_actual.min())  # Normalisiert durch Range
    mape = np.mean(np.abs((test_actual - test_forecast) / test_actual)) * 100
    pearson_corr, _ = pearsonr(test_actual, test_forecast)

    print(f"\n   Validierungsmetriken (Back-Testing auf letzten 24h):")
    print(f"   - NRMSE (Normalized RMSE): {nrmse:.4f}")
    print(f"   - RMSE (Root Mean Squared Error): {rmse:.4f} kW")
    print(f"   - MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"   - PEARSON (Korrelationskoeffizient): {pearson_corr:.4f}")

    # Top 5 Modelle
    print("\n   Top 5 Modelle nach SMAPE:")
    top_models = validation_results.nsmallest(5, "smape")[
        ["Model", "smape", "mae", "rmse"]
    ]
    for idx, row in top_models.iterrows():
        print(
            f"   {row['Model']:25s} - SMAPE: {row['smape']:6.2f}, MAE: {row['mae']:6.2f}, RMSE: {row['rmse']:6.2f}"
        )

    print("\n   Vorhersage (erste 24 Stunden):")
    print(prediction.forecast.head(24))

    print("\n   Obere Grenze des Konfidenzintervalls (erste 10 Stunden):")
    print(prediction.upper_forecast.head(10))

    print("\n   Untere Grenze des Konfidenzintervalls (erste 10 Stunden):")
    print(prediction.lower_forecast.head(10))

    # Statistiken der Vorhersage
    forecasted_daily_consumption = prediction.forecast[
        "power_load_kw"
    ].sum()  # Stündlich
    print(f"\n   === VORHERSAGE-STATISTIKEN ===")
    print(f"   Vorhergesagter Tagesverbrauch: {forecasted_daily_consumption:.2f} kWh")
    print(
        f"   Durchschnittliche Last (Vorhersage): {prediction.forecast['power_load_kw'].mean():.3f} kW"
    )
    print(
        f"   Peak-Last (Vorhersage): {prediction.forecast['power_load_kw'].max():.3f} kW um {prediction.forecast['power_load_kw'].idxmax().strftime('%H:%M')}"
    )
    print(
        f"   Min-Last (Vorhersage): {prediction.forecast['power_load_kw'].min():.3f} kW um {prediction.forecast['power_load_kw'].idxmin().strftime('%H:%M')}"
    )

    # Konfidenzintervall-Breite
    ci_width = (
        prediction.upper_forecast["power_load_kw"]
        - prediction.lower_forecast["power_load_kw"]
    ).mean()
    print(f"   Durchschnittliche CI-Breite (95%): {ci_width:.3f} kW")
    print(
        f"   Unsicherheit (CI-Breite / Mittelwert): {(ci_width / prediction.forecast['power_load_kw'].mean()) * 100:.1f}%"
    )

    # 6. Ergebnisse visualisieren
    print("\n6. Erstelle Visualisierung...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Letzte 7 Tage + Vorhersage
    last_week_samples = 24 * 7  # 7 Tage à 24 Stunden
    ax1.plot(
        df.index[-last_week_samples:],
        df["power_load_kw"][-last_week_samples:],
        label="Historische Daten (letzte 7 Tage)",
        color="blue",
        linewidth=1.5,
    )

    # Vorhersage
    ax1.plot(
        prediction.forecast.index,
        prediction.forecast["power_load_kw"],
        label="Vorhersage (nächste 24h)",
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Konfidenzintervall
    ax1.fill_between(
        prediction.forecast.index,
        prediction.lower_forecast["power_load_kw"],
        prediction.upper_forecast["power_load_kw"],
        alpha=0.3,
        color="red",
        label="95% Konfidenzintervall",
    )

    ax1.set_xlabel("Datum und Uhrzeit")
    ax1.set_ylabel("Stromlast (kW)")
    ax1.set_title("Haushalts-Stromlast: Historische Daten und 24h-Vorhersage")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Nur Vorhersage im Detail
    ax2.plot(
        prediction.forecast.index,
        prediction.forecast["power_load_kw"],
        label="Vorhersage",
        color="red",
        linewidth=2,
        marker="o",
        markersize=2,
    )

    ax2.fill_between(
        prediction.forecast.index,
        prediction.lower_forecast["power_load_kw"],
        prediction.upper_forecast["power_load_kw"],
        alpha=0.3,
        color="red",
        label="95% Konfidenzintervall",
    )

    ax2.set_xlabel("Datum und Uhrzeit")
    ax2.set_ylabel("Stromlast (kW)")
    ax2.set_title("Detailansicht: 24h-Stromlast-Vorhersage (stündlich)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig("household_load_forecast.png", dpi=300, bbox_inches="tight")
    print("   Grafik gespeichert als 'household_load_forecast.png'")

    print("\n Fertig!")
