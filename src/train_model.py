import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap

from src.utils import MODELS_DIR, DATA_PROCESSED, RANDOM_STATE, TEST_SIZE
from src.feature_engineering import prepare_features


def evaluate(y_true, y_pred, name: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def train_baseline(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = evaluate(y_test, y_pred, "LinearRegression")
    return model, result


def train_random_forest(X_train, y_train, X_test, y_test, tune=False):
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        }
        rf = RandomForestRegressor(random_state=RANDOM_STATE)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    result = evaluate(y_test, y_pred, "RandomForest")
    return model, result


def train_xgboost(X_train, y_train, X_test, y_test, tune=False):
    if tune:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
        }
        xgb_model = xgb.XGBRegressor(random_state=RANDOM_STATE)
        grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        print(f"  Best params: {grid.best_params_}")
    else:
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE
        )
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    result = evaluate(y_test, y_pred, "XGBoost")
    return model, result


def explain_model(model, X_test, feature_names):
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        print("  SHAP analysis selesai.")
    except Exception as e:
        print(f"  SHAP tidak dapat dijalankan: {e}")


def run_training(df: pd.DataFrame = None, tune: bool = False):
    if df is None:
        df = pd.read_parquet(DATA_PROCESSED / "data_processed.parquet")

    X, y, encoders = prepare_features(df, fit_encoders=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Training models...")
    results = []
    models = {}

    model_lr, res_lr = train_baseline(X_train, y_train, X_test, y_test)
    results.append(res_lr)
    models["linear"] = model_lr

    model_rf, res_rf = train_random_forest(X_train, y_train, X_test, y_test, tune=tune)
    results.append(res_rf)
    models["random_forest"] = model_rf

    model_xgb, res_xgb = train_xgboost(X_train, y_train, X_test, y_test, tune=tune)
    results.append(res_xgb)
    models["xgboost"] = model_xgb

    best = max(results, key=lambda r: r["R2"])
    print(f"\nModel terbaik: {best['model']} (R2={best['R2']:.4f})")

    explain_model(models["xgboost"], X_test, feature_names)
    joblib.dump(models["xgboost"], MODELS_DIR / "model_xgboost.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    print(f"Model tersimpan di {MODELS_DIR}")
    return models, results


if __name__ == "__main__":
    run_training(tune=False)
