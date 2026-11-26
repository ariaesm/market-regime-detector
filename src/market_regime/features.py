import numpy as np
import pandas as pd


def extract_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts statistical and technical financial features without forward-looking data leakage.
    Features:
      - Log Returns (1-day, 5-day)
      - Realized Volatility (10-day, 20-day annualized)
      - Parkinson High-Low Volatility (10-day)
      - Trend Strength (SMA20 / SMA50 - 1)
      - Normalized Average True Range (NATR)
      - Volume Z-Score
      - RSI (14-day)
    """
    data = df.copy()

    # 1. Log Returns
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1))
    data["log_return_5d"] = np.log(data["Close"] / data["Close"].shift(5))

    # 2. Realized Volatility (Annualized with 252 trading days)
    data["realized_vol_20"] = data["log_return"].rolling(window=20).std() * np.sqrt(252)
    data["realized_vol_10"] = data["log_return"].rolling(window=10).std() * np.sqrt(252)

    # 3. Parkinson Volatility (High-Low volatility estimator)
    hl_ratio = np.log(data["High"] / data["Low"]) ** 2
    parkinson_factor = 1.0 / (4.0 * np.log(2.0))
    data["parkinson_vol_10"] = np.sqrt(hl_ratio.rolling(window=10).mean() * parkinson_factor) * np.sqrt(252)

    # 4. Trend Strength (Moving Average Spread)
    sma_20 = data["Close"].rolling(window=20).mean()
    sma_50 = data["Close"].rolling(window=50).mean()
    data["trend_strength"] = (sma_20 - sma_50) / sma_50

    # 5. Normalized Average True Range (NATR)
    prev_close = data["Close"].shift(1)
    tr1 = data["High"] - data["Low"]
    tr2 = (data["High"] - prev_close).abs()
    tr3 = (data["Low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["natr_14"] = (true_range.rolling(window=14).mean() / data["Close"]) * 100.0

    # 6. Volume Z-Score
    vol_mean = data["Volume"].rolling(window=20).mean()
    vol_std = data["Volume"].rolling(window=20).std()
    data["volume_zscore"] = (data["Volume"] - vol_mean) / (vol_std + 1e-8)

    # 7. Relative Strength Index (RSI 14)
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    data["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # Drop warm-up NaN rows (50 days required for SMA50)
    features_clean = data.dropna().copy()
    return features_clean
