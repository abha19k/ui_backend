import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Optional


def forecast_xgb(
    series: pd.Series,
    horizon: int,
    lag: int = 3,
    *,
    params: Optional[Dict] = None
) -> np.ndarray:
    """
    Forecasts a univariate series using an autoregressive XGBoost model.

    Args:
        series (pd.Series): Input time series (historical values).
        horizon (int): Forecasting horizon (number of steps ahead).
        lag (int): Number of past lags to use as features.
        params (dict, optional): XGBoost parameters.

    Returns:
        np.ndarray: Forecasted values for the given horizon.
    """
    params = params or {}
    # Clean and validate history
    s = pd.to_numeric(series, errors="coerce").dropna()
    H = max(1, horizon)
    L = max(1, lag)

    if len(s) < L + 1:
        # Too short series → fallback to last value forecast
        return np.repeat(float(s.iloc[-1] if len(s) else 0.0), H)

    # Build lagged features (autoregression style)
    X, y = [], []
    for t in range(L, len(s)):
        X.append(s.iloc[t-L:t].values)
        y.append(s.iloc[t])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Default params (kept simple like first code)
    common = dict(
        objective="reg:squarederror",
        n_estimators=params.get("n_estimators", 100),
        random_state=params.get("random_state", 42),
        max_depth=params.get("max_depth", 3),
        learning_rate=params.get("learning_rate", 0.1),
        subsample=params.get("subsample", 1.0),
        colsample_bytree=params.get("colsample_bytree", 1.0)
    )

    model = xgb.XGBRegressor(**common)
    model.fit(X, y)

    # Recursive forecasting
    last_obs = s.iloc[-L:].values.copy()
    preds = []
    for _ in range(H):
        x_input = last_obs.reshape(1, -1).astype(np.float32)
        yhat = float(model.predict(x_input)[0])
        preds.append(yhat)
        last_obs = np.roll(last_obs, -1)
        last_obs[-1] = yhat

    return np.asarray(preds, dtype=float)
