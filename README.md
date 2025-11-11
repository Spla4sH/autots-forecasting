# AutoTS Household Electricity Load Forecasting

Dieses Projekt nutzt die AutoTS-Bibliothek für automatisiertes Time-Series-Forecasting von Stromverbrauch auf Haushaltsebene.

## Features

- **Household Load Forecasting**: Realistische Simulation von Haushaltsstromverbrauch mit Tages- und Wochenmustern
- **Umfassende Metriken**: SMAPE, MAE, RMSE, MAPE, Containment
- **Rolling Window**: Wöchentliches Retraining für produktionsnahe Szenarien
- **Hyperparameter Optimization**: Optuna-Integration für automatisches Tuning

## Installation

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install autots pandas numpy matplotlib optuna joblib
```

## Skripte

### autots_example.py
Hauptbeispiel mit Household Load Forecasting:
- Generiert 3 Monate stündliche Daten
- Forecast-Horizont: 24 Stunden
- Zeigt Top-5-Modelle und detaillierte Metriken
- Visualisierungen: 7-Tage-Historie + 24h-Forecast

```bash
python autots_example.py
```

### autots_rolling_window.py
Rolling Window mit wöchentlichem Retraining:
- Initial Training: 8 Wochen
- Retraining-Intervall: 1 Woche
- Tracking von Metriken über Zeit
- Export als CSV

```bash
python autots_rolling_window.py
```

### autots_hpo_optuna.py
Hyperparameter-Optimierung mit Optuna:
- Optimiert 7 AutoTS-Framework-Parameter
- 15 Trials mit TPE-Sampler
- Exportiert beste Parameter als JSON
- 4 Visualisierungsplots

```bash
python autots_hpo_optuna.py
```

## Datenmerkmale

Die generierten Haushaltsdaten enthalten:
- Basis-Last: ~0.5 kW
- Morgen-Peak: 6-9 Uhr
- Abend-Peak: 18-22 Uhr
- Wochenend-Muster
- Saisonale Variation (Winter-Heizung, Sommer-Kühlung)

## Metriken

- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **MAE**: Mean Absolute Error (kW)
- **RMSE**: Root Mean Squared Error (kW)
- **MAPE**: Mean Absolute Percentage Error
- **Containment**: Prozentsatz innerhalb des Konfidenzintervalls

## Technologie-Stack

- AutoTS 0.6.21
- Python 3.13.9
- Pandas 2.3.3
- NumPy 2.3.4
- Matplotlib 3.10.7
- Optuna (für HPO)
