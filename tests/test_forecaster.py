import numpy as np
from market_regime.data import generate_synthetic_market_data
from market_regime.features import extract_market_features
from market_regime.clustering import RegimeClusterer
from market_regime.forecaster import RegimeForecaster


def test_regime_forecaster():
    df = extract_market_features(generate_synthetic_market_data(n_days=400, seed=42))
    clusterer = RegimeClusterer(n_regimes=3, random_state=42)
    regimes, _, _ = clusterer.fit_predict(df)

    forecaster = RegimeForecaster(random_state=42)
    metrics, y_test, y_pred = forecaster.train_and_evaluate(df, regimes, train_ratio=0.7)

    assert len(y_test) == len(y_pred)
    assert metrics["test_accuracy"] > 0.40  # Must comfortably beat 33% random baseline
    assert metrics["test_macro_f1"] > 0.35
    assert len(metrics["feature_importances"]) == len(RegimeForecaster.FEATURE_COLS)
