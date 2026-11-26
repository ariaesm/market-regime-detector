import numpy as np
import pandas as pd
from typing import Dict, Any


def calculate_performance_metrics(returns: np.ndarray, risk_free_rate: float = 0.03) -> Dict[str, float]:
    """
    Computes standard quantitative investment metrics for a return series.
    """
    if len(returns) == 0:
        return {}

    cum_return = float(np.prod(1.0 + returns) - 1.0)
    n_days = len(returns)
    cagr = float((1.0 + cum_return) ** (252.0 / max(1, n_days)) - 1.0)
    ann_vol = float(np.std(returns) * np.sqrt(252))

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = returns - daily_rf
    sharpe = float(np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(252))

    # Cumulative peak and drawdown
    equity_curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - peak) / peak
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    win_rate = float(np.mean(returns > 0))

    return {
        "cumulative_return_pct": round(cum_return * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "annualized_volatility_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "win_rate_pct": round(win_rate * 100.0, 2),
    }


def backtest_regime_strategy(
    test_returns: np.ndarray,
    predicted_regimes: np.ndarray,
    transaction_cost_bps: float = 5.0,
) -> Dict[str, Any]:
    """
    Simulates a dynamic regime-switching asset allocation on the test set:
      - Regime 2 (Bull): 100% Equity Exposure (weight = 1.0)
      - Regime 1 (Sideways): 50% Equity Exposure (weight = 0.5)
      - Regime 0 (Bear/Panic): 0% Equity Exposure (weight = 0.0, Cash Preservation)
    Accounts for realistic transaction costs when switching positions.
    """
    assert len(test_returns) == len(predicted_regimes), "Length mismatch between returns and regimes"

    # Map regimes to target equity weights
    regime_weights = {
        0: 0.0,  # Bear -> 100% Cash
        1: 0.5,  # Sideways -> 50% Cash / 50% Equity
        2: 1.0,  # Bull -> 100% Equity
    }

    positions = np.array([regime_weights.get(r, 0.0) for r in predicted_regimes])
    # Shift position by 1 day to prevent lookahead bias (execute at next day open)
    executed_positions = np.roll(positions, 1)
    executed_positions[0] = positions[0]

    # Transaction costs on position changes
    position_changes = np.abs(np.diff(executed_positions, prepend=executed_positions[0]))
    cost_factor = (transaction_cost_bps / 10000.0) * position_changes

    strategy_returns = executed_positions * test_returns - cost_factor
    benchmark_returns = test_returns

    strat_metrics = calculate_performance_metrics(strategy_returns)
    bench_metrics = calculate_performance_metrics(benchmark_returns)

    return {
        "strategy": strat_metrics,
        "benchmark_buy_and_hold": bench_metrics,
        "alpha_pct": round(strat_metrics["cumulative_return_pct"] - bench_metrics["cumulative_return_pct"], 2),
        "drawdown_reduction_pct": round(abs(bench_metrics["max_drawdown_pct"]) - abs(strat_metrics["max_drawdown_pct"]), 2),
    }
