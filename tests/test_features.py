from market_regime.data import generate_synthetic_market_data
from market_regime.features import extract_market_features


def test_extract_market_features():
    raw_df = generate_synthetic_market_data(n_days=300, seed=42)
    features = extract_market_features(raw_df)

    # 49 days warm-up dropped for SMA50
    assert len(features) == 251
    assert not features.isnull().any().any()

    expected_cols = [
        "log_return",
        "log_return_5d",
        "realized_vol_20",
        "realized_vol_10",
        "parkinson_vol_10",
        "trend_strength",
        "natr_14",
        "volume_zscore",
        "rsi_14",
    ]
    for col in expected_cols:
        assert col in features.columns

    # Financial logic checks
    assert (features["realized_vol_20"] >= 0).all()
    assert (features["parkinson_vol_10"] >= 0).all()
    assert (features["rsi_14"] >= 0).all() and (features["rsi_14"] <= 100).all()
