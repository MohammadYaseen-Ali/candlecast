from __future__ import annotations
import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]

def load_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise ValueError("CSV must contain a datetime column.")
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    for c in PRICE_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"CSV must contain {c}.")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["datetime"] + PRICE_COLUMNS)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    # Basic OHLC sanity checks; suspicious data should be investigated, not silently fixed.
    bad = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1)) |
        (df["low"] > df[["open", "close", "high"]].min(axis=1)) |
        (df[PRICE_COLUMNS] <= 0).any(axis=1)
    )
    if bad.any():
        print(f"Warning: dropping {int(bad.sum())} rows with invalid OHLC values.")
        df = df.loc[~bad].copy()
    return df.reset_index(drop=True)

def make_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Causal features only: each row uses that candle and earlier candles."""
    d = candles.copy().sort_values("datetime").reset_index(drop=True)
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    eps = 1e-12
    x = pd.DataFrame(index=d.index)

    # Scale price-derived features to avoid dependence on nominal price level.
    x["body_pct"] = (c - o) / (o + eps)
    x["range_pct"] = (h - l) / (c + eps)
    x["upper_wick_pct"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (c + eps)
    x["lower_wick_pct"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (c + eps)
    x["close_position"] = (c - l) / ((h - l) + eps)

    log_return = np.log(c).diff()
    for lag in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64]:
        x[f"log_return_{lag}"] = np.log(c / c.shift(lag))
    for win in [4, 8, 16, 32, 64]:
        x[f"return_mean_{win}"] = log_return.rolling(win).mean()
        x[f"return_std_{win}"] = log_return.rolling(win).std()
        x[f"range_mean_{win}"] = x["range_pct"].rolling(win).mean()
        x[f"body_mean_{win}"] = x["body_pct"].rolling(win).mean()

    for span in [9, 21, 50]:
        ema = c.ewm(span=span, adjust=False).mean()
        x[f"ema_gap_{span}"] = (c - ema) / (c + eps)

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + eps)
    x["rsi_14"] = 100 - (100 / (1 + rs))

    prev_close = c.shift(1)
    true_range = pd.concat([
        h - l, (h - prev_close).abs(), (l - prev_close).abs()
    ], axis=1).max(axis=1)
    x["atr_pct_14"] = true_range.rolling(14).mean() / (c + eps)

    # Calendar features are based on the candle's UTC timestamp.
    dt = pd.to_datetime(d["datetime"], utc=True)
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    x["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    x["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    x["weekday_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    x["weekday_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    # Forex volume is often absent or broker-specific. Use it only if supplied.
    if "volume" in d.columns and d["volume"].notna().sum() > 100:
        vol = pd.to_numeric(d["volume"], errors="coerce")
        x["volume_change"] = np.log1p(vol.clip(lower=0)).diff()
        x["volume_z_32"] = (vol - vol.rolling(32).mean()) / (vol.rolling(32).std() + eps)

    x = x.replace([np.inf, -np.inf], np.nan)
    return x

def make_training_table(candles: pd.DataFrame):
    features = make_features(candles)
    # Direction of the next 3 completed candles, relative to each candle's own open.
    future_bullish = (candles["close"] > candles["open"]).astype(float)
    labels = pd.DataFrame({
        "y_1": future_bullish.shift(-1),
        "y_2": future_bullish.shift(-2),
        "y_3": future_bullish.shift(-3),
    })
    table = pd.concat([candles[["datetime"]], features, labels], axis=1)
    table = table.dropna().reset_index(drop=True)
    feature_cols = list(features.columns)
    return table, feature_cols
