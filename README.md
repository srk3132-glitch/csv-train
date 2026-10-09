# Vehicle Telemetry Speed Models (14 files)

Fourteen regression models, one per telemetry file. Each predicts vehicle speed (`speed_kmh`) from the other sensor readings in the same file.

## Folder layout

All files sit in one folder, no subfolders:

```
README.md
model_01.py ... model_14.py      one training script per file
telemetry_01.csv ... telemetry_14.csv   the data each script reads (same name number)
```

## How to run one model

```bash
pip install pandas scikit-learn joblib
python model_01.py
```

Each script reads the CSV beside it, trains on 80% of rows, prints held-out R², MAE and RMSE, then refits on all rows and saves a `.joblib` file in the same folder.

## Results per model

| Model | Data file | Rows | Algorithm | Test R² | MAE (km/h) | RMSE (km/h) | Incident rate |
|---|---|---|---|---|---|---|---|
| model_01 | telemetry_01.csv | 400 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 68.8% |
| model_02 | telemetry_02.csv | 300 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 53.3% |
| model_03 | telemetry_03.csv | 300 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 58.3% |
| model_04 | telemetry_04.csv | 300 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 50.0% |
| model_05 | telemetry_05.csv | 300 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 50.0% |
| model_06 | telemetry_06.csv | 400 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 75.0% |
| model_07 | telemetry_07.csv | 350 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 64.3% |
| model_08 | telemetry_08.csv | 300 | LinearRegression | 1.0000 | 0.0012 | 0.0035 | 58.3% |
| model_09 | telemetry_09.csv | 400 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 75.0% |
| model_10 | telemetry_10.csv | 250 | HuberRegressor | 1.0000 | 0.0000 | 0.0000 | 70.0% |
| model_11 | telemetry_11.csv | 400 | HuberRegressor | 1.0000 | 0.0056 | 0.0120 | 75.0% |
| model_12 | telemetry_12.csv | 300 | HuberRegressor | 1.0000 | 0.0162 | 0.1252 | 50.0% |
| model_13 | telemetry_13.csv | 300 | AdaBoost | 1.0000 | 0.0333 | 0.2582 | 50.0% |
| model_14 | telemetry_14.csv | 350 | LinearRegression | 1.0000 | 0.0000 | 0.0000 | 57.1% |

Metrics use a random 20% held-out split (seed 42). MAE and RMSE are in km/h. Incident rate is the share of rows with `incident_flag = 1`.

## What each model is good for, and why

- **Linear regression (models 01–09, 14):** The best fit for these files. Speed follows rpm (and gear, where present) in a fixed linear way, so a linear model is exact, very fast, and fully explainable: each coefficient tells you how much a sensor moves predicted speed. Use it when you need a simple, transparent predictor. Models 01–07, 09 and 14 all show MAE of 0.0 km/h on held-out rows (rounded to four decimals).
- **Huber regression (models 10–12):** Same linear form, but it down-weights extreme readings. Choose it when a file has spikes or outliers (for example sudden acceleration or braking) and you still want a linear model. Model 11 has a small non-zero error (MAE 0.0056 km/h) and model 12 has a larger RMSE (0.125 km/h) caused by a few large misses.
- **AdaBoost (model 13):** An ensemble of decision trees, and the only non-linear winner. It has the largest error of the set (MAE 0.033 km/h, RMSE 0.258 km/h), so it is less precise than the linear models on this file. Use it only if the relationship in a future file stops being linear, since it is harder to interpret and slower to retrain.

## Limitations

- **An R² of 1.0 means the relationship is exact in this data.** Speed is almost a fixed function of rpm here (about 0.045 × rpm, with gear adjusting it). The models have learned that rule. They will not necessarily score this high on a different vehicle, road, or driving style, so validate on new data before relying on them.
- **Speed is an easy target because rpm is an input.** Harder targets, such as acceleration or yaw rate, give much lower and less consistent scores across files.
- **Features differ per file.** Only columns that vary within a file are used, so constant sensors (`oil_bar`, `battery_v`, `tyre_*` in most files, `lat`, `lon`) are dropped. `time_s` is excluded. `incident_flag` is included as an input in every model, which is a small leak risk if the flag is only known after the event. The exact feature list for each model is in its `FEATURES` line.
- **The split is random across rows, not by time.** Telemetry rows are sequential and neighbouring rows are alike, so the held-out scores are likely optimistic. A time-based split would be a stricter test.

## Feature lists

- Every model: `rpm`, `heading_deg`, `accel_x_g`, `accel_y_g`, `yaw_rate_dps`, `engine_temp_c`, `fuel_pct`, `incident_flag`.
- Models 04, 12, 13: also `gear`.
- Models 06, 08, 09: also `brake_pct`.
- Model 06: also `tyre_fl`.
