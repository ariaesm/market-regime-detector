import numpy as np
from market_regime.backtest import calculate_performance_metrics, backtest_regime_strategy


def test_performance_metrics():
    # Constant 1% daily return over 100 days
    returns = np.full(100, 0.01)
    metrics = calculate_performance_metrics(returns)

    assert metrics["cumulative_return_pct"] > 100.0
    assert metrics["win_rate_pct"] == 100.0
    assert metrics["max_drawdown_pct"] == 0.0


def test_backtest_regime_strategy():
    returns = np.array([0.02, -0.01, 0.03, -0.05, 0.01, 0.02])
    regimes = np.array([2, 1, 2, 0, 1, 2])

    result = backtest_regime_strategy(returns, regimes)

    assert "strategy" in result
    assert "benchmark_buy_and_hold" in result
    assert "alpha_pct" in result
    assert "drawdown_reduction_pct" in result
