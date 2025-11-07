# forecast_7d_multi_models_backtest_with_plots_pi.py
import os, math, warnings, re
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# CONFIG (DAILY, 7D FORWARD)
# =========================
HISTORY_FILE      = "History_Daily.csv"   # ProductID, ChannelID, LocationID, StartDate, EndDate, Qty

# Outputs (CSV)
OUT_BACKTEST_CSV  = "Backtest_Daily_Predictions_ByModel.csv"
OUT_SUMMARY_CSV   = "Model_Selection_Summary_Daily_ByCombo.csv"
OUT_FORWARD_CSV   = "Future7D_Forecasts_BestModel.csv"

# Plots
PLOTS_DIR_BACKTEST = os.path.join("plots", "backtest_daily")
PLOTS_DIR_FORWARD  = os.path.join("plots", "history_plus_forecast_daily")

# Feature engineering
LAG_WINDOW        = 7          # use last 7 days as basic lags
USE_SEASONAL_L12  = True       # interpreted as "use seasonal lag" (we’ll use SEASONAL_LAG below)
ADD_MONTH_DUMMIES = False      # usually less relevant for only 7 days

# Seasonal config
SEASONAL_LAG      = 7          # daily seasonal lag (weekly pattern)
SEASONAL_PERIOD   = 7          # seasonal period for seasonal_naive

# Backtest & selection
MIN_TRAIN_POINTS  = 30         # require at least ~1 month of history
CV_SPLITS_MIN     = 3
METRIC            = "WMAPE"    # "WMAPE" or "MAE" for model selection
FAST_MODE         = True
CV_STRIDE         = 2          # evaluate every 2nd step during backtest to speed up

# Prefer SNaive unless best model beats it by >= this fraction (5% = 0.05) on selection metric
SNAIVE_PREFERENCE_MARGIN = 0.05

# Forward horizon
FORWARD_HORIZON   = 7          # next 7 days

# Plotting controls
MAX_PLOTS         = 250        # safety cap for large runs
DPI               = 140
FIGSIZE           = (10, 5)

# =========================
# HELPERS
# =========================
def parse_dt_exact(s): 
    return pd.to_datetime(s, dayfirst=True, errors="coerce")

def parse_qty_series(q: pd.Series) -> pd.Series:
    s = q.astype(str).str.replace("\u00A0","",regex=False).str.replace(" ","",regex=False).str.strip()
    comma = s.str.contains(",", regex=False)
    dot   = s.str.contains(r"\.", regex=True)
    s = s.where(~(comma & ~dot), s.str.replace(",", ".", regex=False))   # decimal comma -> dot
    s = s.str.replace(",", "", regex=False)                              # thousands sep
    return pd.to_numeric(s, errors="coerce")

def wmape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    mask = (~np.isnan(yt)) & (~np.isnan(yp)) & (yt != 0)
    if mask.sum() == 0: return np.inf
    return np.abs(yt[mask] - yp[mask]).sum() / np.abs(yt[mask]).sum()

def mae(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float); yp = np.asarray(y_pred, dtype=float)
    mask = (~np.isnan(yt)) & (~np.isnan(yp))
    if mask.sum() == 0: return np.inf
    return np.abs(yt[mask] - yp[mask]).mean()

def score_metric(y_true, y_pred, metric: str):
    return wmape(y_true, y_pred) if metric.upper()=="WMAPE" else mae(y_true, y_pred)

def clamp_nonneg(x):
    x = np.asarray(x, dtype=float)
    x[np.isnan(x)] = 0.0
    return np.maximum(0.0, x)

def safe_key(*parts):
    s = " | ".join(str(p) for p in parts)
    s = re.sub(r"[\\/*?:\"<>|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Design matrix for log target; returns X, y, aligned index, and column names
def make_design_matrix(dates: pd.DatetimeIndex, y_log: pd.Series,
                       lag_window: int, use_l12: bool, add_months: bool):
    df = pd.DataFrame({"y": y_log.values}, index=dates)
    for k in range(1, lag_window+1):
        df[f"lag{k}"] = df["y"].shift(k)
    # "Seasonal" lag based on SEASONAL_LAG (here 7 days)
    if use_l12 and SEASONAL_LAG > 0:
        df["lag_season"] = df["y"].shift(SEASONAL_LAG)
    if add_months:
        df["month"] = [d.month for d in df.index]
        months = pd.get_dummies(df["month"], prefix="m", dtype=float)
        df = pd.concat([df, months], axis=1)
    df = df.dropna()
    y_out = df["y"].values.astype(float)
    cols = [c for c in df.columns if c not in ("y","month")]
    X_out = df[cols].values.astype(float)
    return X_out, y_out, df.index, cols

# Seasonal naive with configurable period (here 7 days)
def seasonal_naive(series: pd.Series, horizon: int, season_len: int = SEASONAL_PERIOD):
    vals = series.values.astype(float)
    out = []
    for h in range(1, horizon + 1):
        if len(vals) >= season_len:
            idx = len(vals) - season_len + (h - 1)
            if idx < len(vals):
                out.append(vals[idx])
            else:
                out.append(out[-season_len] if len(out) >= season_len else vals[-1])
        else:
            out.append(vals[-1] if len(vals) else 0.0)
    return np.array(out, dtype=float)

def next_feature_row_linear_space(hist_lin: pd.Series, next_dt: pd.Timestamp,
                                  colnames: list[str], lag_window: int, use_l12: bool, add_months: bool):
    feats = {}
    # simple lags
    for k in range(1, lag_window+1):
        val = hist_lin.iloc[-k] if len(hist_lin) >= k else np.nan
        feats[f"lag{k}"] = np.log1p(max(0.0, float(val))) if pd.notna(val) else 0.0
    # seasonal lag
    if use_l12 and SEASONAL_LAG > 0:
        val_s = hist_lin.iloc[-SEASONAL_LAG] if len(hist_lin) >= SEASONAL_LAG else np.nan
        feats["lag_season"] = np.log1p(max(0.0, float(val_s))) if pd.notna(val_s) else 0.0
    if add_months:
        m = int(next_dt.month)
        for j in range(1, 13):
            feats[f"m_{j}"] = 1.0 if m == j else 0.0
    row = [float(feats.get(c, 0.0)) for c in colnames]
    return np.array(row, dtype=float).reshape(1, -1)

# =========================
# MODELS
# =========================
def get_model_registry():
    registry = {}

    # XGBoost (optional)
    try:
        import xgboost as xgb
        def make_xgb():
            return xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=700, max_depth=4, learning_rate=0.03,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                tree_method="hist", grow_policy="lossguide", n_jobs=-1,
                random_state=42
            )
        registry["xgb"] = make_xgb
    except Exception:
        pass

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    registry["rf"] = lambda: RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, random_state=42
    )

    # Gradient Boosting (sklearn)
    from sklearn.ensemble import GradientBoostingRegressor
    registry["gbr"] = lambda: GradientBoostingRegressor(
        n_estimators=600, max_depth=3, learning_rate=0.05,
        subsample=0.9, random_state=42
    )

    # Ridge (linear baseline)
    from sklearn.linear_model import Ridge
    registry["ridge"] = lambda: Ridge(alpha=1.0, random_state=42)

    return registry

# =========================
# BACKTEST (rolling-origin) — ALIGNED
# =========================
def rolling_backtest(series: pd.Series, metric: str, fast_stride: int = 1):
    # Build supervised on LOG target
    y_log = np.log1p(np.maximum(0.0, series.values))
    X_all, y_all, idx_all, colnames = make_design_matrix(
        series.index,
        pd.Series(y_log, index=series.index),
        LAG_WINDOW, USE_SEASONAL_L12, ADD_MONTH_DUMMIES
    )
    if X_all is None or len(y_all) < 3:
        return {}, {}, (None, None, None, None)

    # predict y_all[i] using features ending at i-1
    idxs = list(range(1, len(y_all)))
    if FAST_MODE and fast_stride > 1:
        idxs = idxs[::fast_stride]

    registry = get_model_registry()
    preds_dict = {name: {"dates": [], "preds": []} for name in registry.keys()}
    preds_dict["snaive"] = {"dates": [], "preds": []}

    truths_aligned = []

    # Walk-forward
    for i in idxs:
        raw_points_before_target = LAG_WINDOW + i
        if raw_points_before_target < MIN_TRAIN_POINTS:
            continue

        X_tr, y_tr = X_all[:i], y_all[:i]
        split = max(1, int(0.1 * len(X_tr))) if len(X_tr) > 12 and "xgb" in registry else 0
        X_fit, y_fit = (X_tr[:-split], y_tr[:-split]) if split > 0 else (X_tr, y_tr)
        X_val, y_val = (X_tr[-split:], y_tr[-split:]) if split > 0 else (None, None)

        for name, maker in registry.items():
            try:
                model = maker()
                if name == "xgb" and split > 0:
                    try:
                        model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                    except TypeError:
                        model.fit(X_tr, y_tr)
                else:
                    model.fit(X_tr, y_tr)

                p_log = float(model.predict(X_all[i-1].reshape(1, -1))[0])
                p_lin = max(0.0, math.expm1(p_log))
            except Exception:
                p_lin = np.nan

            preds_dict[name]["dates"].append(idx_all[i])
            preds_dict[name]["preds"].append(p_lin)

        # Seasonal naive on history aligned to supervised index
        hist_upto_lin = series.reindex(idx_all[:i])
        sn = seasonal_naive(hist_upto_lin, 1)[0]
        preds_dict["snaive"]["dates"].append(idx_all[i])
        preds_dict["snaive"]["preds"].append(float(sn))

        # Ground truth at the supervised target (back-transform)
        truths_aligned.append(float(np.expm1(y_all[i])))

    # Scores (vs aligned truths)
    scores = {}
    for name, bucket in preds_dict.items():
        if len(bucket["preds"]) == 0:
            scores[name] = {"WMAPE": np.inf, "MAE": np.inf, "splits": 0}
            continue
        s_w = wmape(truths_aligned, bucket["preds"])
        s_m = mae(truths_aligned, bucket["preds"])
        scores[name] = {"WMAPE": s_w, "MAE": s_m, "splits": len(bucket["preds"])}

    return preds_dict, scores, (X_all, y_all, idx_all, colnames)

# =========================
# FINAL FIT & FORWARD FORECAST
# =========================
def forward_forecast_best(series: pd.Series, best_model_name: str, horizon: int = FORWARD_HORIZON):
    registry = get_model_registry()
    if best_model_name == "snaive" or best_model_name not in registry:
        return seasonal_naive(series, horizon)

    y_log = np.log1p(np.maximum(0.0, series.values))
    X_all, y_all, idx_all, cols = make_design_matrix(
        series.index,
        pd.Series(y_log, index=series.index),
        LAG_WINDOW, USE_SEASONAL_L12, ADD_MONTH_DUMMIES
    )
    if X_all is None or len(y_all) < LAG_WINDOW + 1:
        last_val = float(series.iloc[-1]) if len(series) else 0.0
        return np.repeat(last_val, horizon)

    model = registry[best_model_name]()
    if best_model_name == "xgb" and len(X_all) > 12:
        split = max(1, int(0.1 * len(X_all)))
        X_fit, y_fit = X_all[:-split], y_all[:-split]
        X_val, y_val = X_all[-split:], y_all[-split:]
        try:
            model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        except TypeError:
            model.fit(X_all, y_all)
    else:
        model.fit(X_all, y_all)

    hist_lin = series.copy()
    out = []
    for _ in range(horizon):
        # Daily: next calendar day
        next_dt = hist_lin.index.max() + pd.Timedelta(days=1)
        xrow = next_feature_row_linear_space(hist_lin, next_dt, cols, LAG_WINDOW, USE_SEASONAL_L12, ADD_MONTH_DUMMIES)
        p_log = float(model.predict(xrow)[0])
        p_lin = max(0.0, math.expm1(p_log))
        out.append(p_lin)
        hist_lin = pd.concat([hist_lin, pd.Series([p_lin], index=[next_dt])])

    return clamp_nonneg(out)

# =========================
# PLOTTING (Aligned + Scores + PI bands)
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_backtest_combo(pid, chanid, locid, series, preds_dict, out_dir=PLOTS_DIR_BACKTEST):
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    key = safe_key(pid, chanid, locid)
    path = os.path.join(out_dir, f"backtest_{key}.png")

    # union of prediction dates (already aligned to supervised index)
    all_dates = sorted({d for b in preds_dict.values() for d in b["dates"]})
    y_actual = series.reindex(all_dates).astype(float).values  # aligned truths

    plt.figure(figsize=FIGSIZE, dpi=DPI)
    if len(all_dates):
        plt.plot(all_dates, y_actual, label="Actual", linewidth=2, marker="o", markersize=3)

    for model_name, bucket in preds_dict.items():
        if len(bucket["dates"]) == 0:
            continue
        # align predictions to the same date order
        idx_map = {d:i for i,d in enumerate(all_dates)}
        preds = np.full(len(all_dates), np.nan)
        for d, p in zip(bucket["dates"], bucket["preds"]):
            if d in idx_map:
                preds[idx_map[d]] = p
        # scores for legend
        m_w = wmape(y_actual, preds)
        m_m = mae(y_actual, preds)
        label = f"{model_name} (WMAPE {m_w:.2%}, MAE {m_m:.0f})"
        plt.plot(all_dates, preds, label=label, linewidth=1, marker=".", markersize=2)

    plt.title(f"Backtest (Aligned, Daily) | {pid} / {chanid} / {locid}")
    plt.xlabel("Date"); plt.ylabel("Qty")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", ncol=1, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(path); plt.close()
    return path

def empirical_pi_from_backtest(preds, actuals, q_low=0.1, q_high=0.9):
    """Return additive P10/P90 residual quantiles from backtest residuals."""
    preds = np.asarray(preds, float); actuals = np.asarray(actuals, float)
    mask = (~np.isnan(preds)) & (~np.isnan(actuals))
    if mask.sum() < 6:
        return 0.0, 0.0
    resid = actuals[mask] - preds[mask]
    return np.quantile(resid, q_low), np.quantile(resid, q_high)

def plot_history_plus_forecast(pid, chanid, locid, hist_df, fut_df, out_dir=PLOTS_DIR_FORWARD,
                               backtest_df_path=OUT_BACKTEST_CSV):
    import matplotlib.pyplot as plt
    ensure_dir(out_dir)
    key = safe_key(pid, chanid, locid)
    path = os.path.join(out_dir, f"history_forecast_{key}.png")

    # History
    sub_hist = hist_df[(hist_df["ProductID"]==pid) & (hist_df["ChannelID"]==chanid) & (hist_df["LocationID"]==locid)]
    ser = (sub_hist.set_index("StartDate")["Qty"].sort_index())

    # Forecasts (best model rows)
    sub_fut = fut_df[(fut_df["ProductID"]==pid) & (fut_df["ChannelID"]==chanid) & (fut_df["LocationID"]==locid)]

    # Pull backtest residuals for the best model to estimate PI
    p10_add, p90_add = 0.0, 0.0
    best_model = None
    if len(sub_fut):
        best_model = sub_fut["Model"].iloc[0]
        if os.path.exists(backtest_df_path):
            bt = pd.read_csv(backtest_df_path)
            bt["Date"] = parse_dt_exact(bt["Date"])
            bt_sub = bt[(bt["ProductID"]==pid) & (bt["ChannelID"]==chanid) & (bt["LocationID"]==locid) & (bt["Model"]==best_model)]
            if not bt_sub.empty:
                preds = bt_sub.sort_values("Date")["Pred"].values
                acts  = bt_sub.sort_values("Date")["Actual"].values
                p10_add, p90_add = empirical_pi_from_backtest(preds, acts)

    # Plot
    plt.figure(figsize=FIGSIZE, dpi=DPI)

    if len(ser):
        plt.plot(ser.index, ser.values, label="History (Actual)", linewidth=2)

    if len(sub_fut):
        fut_dates = parse_dt_exact(sub_fut["StartDate"])
        fut_vals  = sub_fut["Forecast Qty"].values.astype(float)
        plt.plot(fut_dates, fut_vals, label=f"Forecast 7D ({best_model})", linewidth=2)

        # Empirical P10/P90 bands (additive)
        lo = fut_vals + p10_add
        hi = fut_vals + p90_add
        plt.fill_between(fut_dates, lo, hi, alpha=0.18, label="Empirical P10–P90")

    plt.title(f"History + 7D Forecast | {pid} / {chanid} / {locid}")
    plt.xlabel("Date"); plt.ylabel("Qty")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path); plt.close()
    return path

# =========================
# CORE
# =========================
def run():
    if not os.path.exists(HISTORY_FILE):
        raise FileNotFoundError(f"{HISTORY_FILE} not found.")

    raw = pd.read_csv(HISTORY_FILE)
    for c in ["ProductID","LocationID","ChannelID"]:
        if c in raw.columns:
            raw[c] = raw[c].astype(str).str.replace("\u00A0","",regex=False).str.strip()
    raw["StartDate"] = parse_dt_exact(raw["StartDate"])
    raw["EndDate"]   = parse_dt_exact(raw["EndDate"])
    raw["Qty"]       = parse_qty_series(raw["Qty"])

    hist = (raw.dropna(subset=["StartDate"])
              .groupby(["ProductID","LocationID","ChannelID","StartDate","EndDate"], as_index=False)["Qty"]
              .sum())

    if hist.empty:
        print("[WARN] No valid history.")
        return {}

    # per-combo series
    series_all = (hist
                  .set_index(["ProductID","LocationID","ChannelID","StartDate"])["Qty"]
                  .sort_index())

    combos = hist[["ProductID","LocationID","ChannelID"]].drop_duplicates()

    backtest_rows = []
    summary_rows  = []
    forward_rows  = []
    plot_count    = 0

    # Precompute friendly history (for plotting)
    hist_for_plot = hist.copy()

    for _, rr in combos.iterrows():
        pid, locid, chanid = rr["ProductID"], rr["LocationID"], rr["ChannelID"]
        try:
            sub = series_all.loc[(pid, locid, chanid)].copy().sort_index()
        except KeyError:
            continue
        if len(sub) < MIN_TRAIN_POINTS:
            continue

        # ---- Backtest (aligned)
        preds_dict, scores, feat_pack = rolling_backtest(sub, METRIC, CV_STRIDE if FAST_MODE else 1)
        X_all, y_all, idx_all, colnames = feat_pack
        if len(scores)==0 or all(v.get("splits",0)==0 for v in scores.values()):
            continue

        metric_key = "WMAPE" if METRIC.upper()=="WMAPE" else "MAE"
        best_model = min(scores.items(), key=lambda kv: kv[1][metric_key])[0]
        best_score = scores[best_model][metric_key]

        # Prefer snaive unless best beats it by >= margin
        if "snaive" in scores and best_model != "snaive":
            snaive_score = scores["snaive"][metric_key]
            if not np.isinf(snaive_score):
                gain = snaive_score - best_score
                if gain < SNAIVE_PREFERENCE_MARGIN * snaive_score:
                    best_model, best_score = "snaive", snaive_score

        # Write backtest rows
        for model_name, bucket in preds_dict.items():
            for dt, pred in zip(bucket["dates"], bucket["preds"]):
                truth = float(sub.reindex([dt]).iloc[0]) if dt in sub.index else np.nan
                ape = (abs(truth - pred) / abs(truth) * 100.0) if (pd.notna(truth) and truth != 0 and pd.notna(pred)) else np.nan
                backtest_rows.append({
                    "ProductID": pid, "ChannelID": chanid, "LocationID": locid,
                    "Date": dt.strftime("%d/%m/%Y"),
                    "Model": model_name,
                    "Pred": float(pred) if pred is not None else np.nan,
                    "Actual": truth,
                    "APE": ape
                })

        # Write summary row
        row_s = {"ProductID": pid, "ChannelID": chanid, "LocationID": locid, "BestModel": best_model, "BestScore": best_score}
        for name, sc in scores.items():
            row_s[f"{name}_WMAPE"] = sc["WMAPE"]
            row_s[f"{name}_MAE"]   = sc["MAE"]
            row_s[f"{name}_splits"]= sc["splits"]
        summary_rows.append(row_s)

        # ---- Forward 7D using best model (daily)
        combo_tab = hist[(hist["ProductID"]==pid) & (hist["LocationID"]==locid) & (hist["ChannelID"]==chanid)]
        last_start = combo_tab["StartDate"].max()

        future_starts = [last_start + pd.Timedelta(days=i+1) for i in range(FORWARD_HORIZON)]
        future_ends   = [dt for dt in future_starts]  # daily: same day

        fut_vals = forward_forecast_best(sub, best_model, FORWARD_HORIZON)
        for f_start, f_end, fc in zip(future_starts, future_ends, fut_vals):
            asof = f_start
            hist_cutoff = asof - pd.Timedelta(days=1)
            forward_rows.append({
                "ProductID": pid, "ChannelID": chanid, "LocationID": locid,
                "Model": best_model,
                "History End Date": hist_cutoff.strftime("%d/%m/%Y"),
                "Forecast Date": asof.strftime("%d/%m/%Y"),
                "StartDate": f_start.strftime("%d/%m/%Y"),
                "EndDate": f_end.strftime("%d/%m/%Y"),
                "Forecast Qty": float(fc),
                "Qty": np.nan
            })

        # ---- Plots
        if plot_count < MAX_PLOTS:
            try:
                _ = plot_backtest_combo(pid, chanid, locid, sub, preds_dict, PLOTS_DIR_BACKTEST)
                plot_count += 1
            except Exception as e:
                print(f"[WARN] Plot backtest failed for {pid}/{chanid}/{locid}: {e}")

        if plot_count < MAX_PLOTS:
            try:
                fut_small = pd.DataFrame([r for r in forward_rows if r["ProductID"]==pid and r["ChannelID"]==chanid and r["LocationID"]==locid])
                _ = plot_history_plus_forecast(pid, chanid, locid, hist_for_plot, fut_small, PLOTS_DIR_FORWARD)
                plot_count += 1
            except Exception as e:
                print(f"[WARN] Plot history+forecast failed for {pid}/{chanid}/{locid}: {e}")

    # ---- Write outputs
    if backtest_rows:
        df_bt = pd.DataFrame(backtest_rows)
        df_bt.to_csv(OUT_BACKTEST_CSV, index=False)
        print(f"[OK] Backtest (daily) written: {OUT_BACKTEST_CSV} | rows={len(df_bt)}")
    else:
        print("[WARN] No backtest rows written (daily).")

    if summary_rows:
        df_sm = pd.DataFrame(summary_rows)
        df_sm.to_csv(OUT_SUMMARY_CSV, index=False)
        print(f"[OK] Summary (daily) written: {OUT_SUMMARY_CSV} | rows={len(df_sm)}")
    else:
        print("[WARN] No summary rows written (daily).")

    if forward_rows:
        df_fw = pd.DataFrame(forward_rows)
        df_fw["_Start_dt"] = parse_dt_exact(df_fw["StartDate"])
        df_fw = df_fw.sort_values(by=["ProductID","ChannelID","LocationID","_Start_dt"]).drop(columns=["_Start_dt"]).reset_index(drop=True)
        df_fw.to_csv(OUT_FORWARD_CSV, index=False)
        print(f"[OK] Forward 7D written: {OUT_FORWARD_CSV} | rows={len(df_fw)}")
    else:
        print("[WARN] No forward rows written (daily).")

    print(f"[OK] Daily plots saved to:\n  - {PLOTS_DIR_BACKTEST}\n  - {PLOTS_DIR_FORWARD}")

    return {
        "backtest_file": OUT_BACKTEST_CSV,
        "summary_file": OUT_SUMMARY_CSV,
        "forward_file": OUT_FORWARD_CSV,
        "plots_backtest_dir": PLOTS_DIR_BACKTEST,
        "plots_history_forecast_dir": PLOTS_DIR_FORWARD
    }

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run()
