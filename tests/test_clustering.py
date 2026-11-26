import numpy as np
from market_regime.data import generate_synthetic_market_data
from market_regime.features import extract_market_features
from market_regime.clustering import RegimeClusterer


def test_regime_clusterer():
    df = extract_market_features(generate_synthetic_market_data(n_days=400, seed=42))
    clusterer = RegimeClusterer(n_regimes=3, random_state=42)
    regimes, probs, stats = clusterer.fit_predict(df)

    assert len(regimes) == len(df)
    assert set(np.unique(regimes)).issubset({0, 1, 2})

    # Probabilities must sum to 1.0 for each row
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # Transition matrix must be valid stochastic matrix (rows sum to 1.0)
    trans = stats["transition_matrix"]
    assert trans.shape == (3, 3)
    np.testing.assert_allclose(trans.sum(axis=1), 1.0, atol=1e-5)

    assert stats["silhouette_score"] > 0.0
