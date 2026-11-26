import numpy as np
import pandas as pd


def generate_synthetic_market_data(
    n_days: int = 1000,
    start_price: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates realistic daily OHLCV market data simulating 3 distinct regimes:
      - Regime 0: Bull market (positive drift mu=0.0008, low volatility sigma=0.010)
      - Regime 1: Bear / Panic selloff (negative drift mu=-0.0015, high volatility sigma=0.030)
      - Regime 2: Sideways / Range-bound (zero drift mu=0.0000, moderate volatility sigma=0.015)
    """
    rng = np.random.default_rng(seed)

    # Transition probability matrix between regimes
    # High persistence (regimes typically last 30-100 trading days)
    transition_matrix = np.array([
        [0.97, 0.01, 0.02],  # From Bull
        [0.03, 0.94, 0.03],  # From Bear/Panic
        [0.03, 0.02, 0.95],  # From Sideways
    ])

    regimes_params = {
        0: {"mu": 0.0008, "sigma": 0.010},  # Bull
        1: {"mu": -0.0015, "sigma": 0.028}, # Bear
        2: {"mu": 0.0000, "sigma": 0.014},  # Sideways
    }

    current_regime = 0
    regime_history = []
    prices = [start_price]
    opens, highs, lows, volumes = [], [], [], []

    date_range = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="B")

    for i in range(n_days):
        # Sample next regime according to transition probabilities
        current_regime = rng.choice(3, p=transition_matrix[current_regime])
        regime_history.append(current_regime)

        p = regimes_params[current_regime]
        daily_return = rng.normal(p["mu"], p["sigma"])
        new_close = max(1.0, prices[-1] * np.exp(daily_return))

        # Intraday High, Low, Open
        intraday_vol = p["sigma"] * rng.uniform(0.6, 1.4)
        open_price = prices[-1] * np.exp(rng.normal(0, p["sigma"] * 0.3))
        high_price = max(open_price, new_close) * (1.0 + abs(rng.normal(0, intraday_vol)))
        low_price = min(open_price, new_close) * (1.0 - abs(rng.normal(0, intraday_vol)))
        volume = int(rng.lognormal(mean=12.0 + current_regime * 0.4, sigma=0.5))

        prices.append(new_close)
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        volumes.append(volume)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices[1:],
            "Volume": volumes,
            "TrueRegime": regime_history,
        },
        index=date_range,
    )
    return df
