"""
Market Regime Detector - Quantitative Machine Learning for Regime Classification and Trading.
"""

from market_regime.data import generate_synthetic_market_data
from market_regime.features import extract_market_features
from market_regime.clustering import RegimeClusterer
from market_regime.forecaster import RegimeForecaster
from market_regime.backtest import backtest_regime_strategy

__version__ = "1.0.0"
__all__ = [
    "generate_synthetic_market_data",
    "extract_market_features",
    "RegimeClusterer",
    "RegimeForecaster",
    "backtest_regime_strategy",
]
