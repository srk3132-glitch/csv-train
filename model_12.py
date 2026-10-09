"""model_12: trains a HuberRegressor model on telemetry_12.csv to predict speed_kmh.

Usage:  python model_12.py
Reads telemetry_12.csv (path below), trains on 80% of rows, reports held-out metrics,
then refits on all rows and saves model_12.joblib next to this script.
"""
import pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA = Path(__file__).resolve().parent / "telemetry_12.csv"
TARGET = "speed_kmh"
FEATURES = ['rpm', 'gear', 'heading_deg', 'accel_x_g', 'accel_y_g', 'yaw_rate_dps', 'engine_temp_c', 'fuel_pct', 'incident_flag']
SEED = 42

def build_model():
    from sklearn.linear_model import HuberRegressor
    return make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000))

if __name__ == "__main__":
    df = pd.read_csv(DATA)
    X, y = df[FEATURES].values, df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = build_model().fit(Xtr, ytr)
    p = model.predict(Xte)
    print("Held-out R2  :", round(r2_score(yte, p), 4))
    print("Held-out MAE :", round(mean_absolute_error(yte, p), 4), "km/h")
    print("Held-out RMSE:", round(mean_squared_error(yte, p) ** 0.5, 4), "km/h")
    final = build_model().fit(X, y)
    joblib.dump({"model": final, "features": FEATURES, "target": TARGET}, Path(__file__).with_suffix(".joblib"))
    print("Saved", Path(__file__).with_suffix(".joblib").name)
