import argparse
import sys
from market_regime.data import generate_synthetic_market_data
from market_regime.features import extract_market_features
from market_regime.clustering import RegimeClusterer
from market_regime.forecaster import RegimeForecaster
from market_regime.backtest import backtest_regime_strategy


def run_pipeline(n_days: int = 1200, seed: int = 42):
    print("=" * 72)
    print("  MARKET REGIME DETECTOR — QUANTITATIVE MACHINE LEARNING PIPELINE")
    print("=" * 72)

    # 1. Ingest Data
    print(f"\n[1/4] Generating & ingesting {n_days} days of multi-regime OHLCV data...")
    raw_df = generate_synthetic_market_data(n_days=n_days, seed=seed)
    print(f"      Initial close: ${raw_df['Close'].iloc[0]:.2f} -> Final close: ${raw_df['Close'].iloc[-1]:.2f}")

    # 2. Extract ML Features
    print("\n[2/4] Engineering financial features (Vol, Momentum, Parkinson, NATR, RSI)...")
    df = extract_market_features(raw_df)
    print(f"      Constructed {len(df)} feature records without lookahead leakage.")

    # 3. Unsupervised GMM Clustering
    print("\n[3/4] Fitting Gaussian Mixture Model (GMM) for latent regime discovery...")
    clusterer = RegimeClusterer(n_regimes=3, random_state=seed)
    regimes, probs, cluster_stats = clusterer.fit_predict(df)
    print(f"      Silhouette Score: {cluster_stats['silhouette_score']:.4f}")
    print(f"      Regime Distribution: {cluster_stats['regime_distribution']}")

    # 4. Supervised Forecasting & Backtest
    print("\n[4/4] Training chronological Random Forest Forecaster (75% Train / 25% Test)...")
    forecaster = RegimeForecaster(random_state=seed)
    fc_metrics, y_test, y_pred = forecaster.train_and_evaluate(df, regimes, train_ratio=0.75)
    print(f"      Out-of-sample Test Accuracy: {fc_metrics['test_accuracy'] * 100.0:.2f}%")
    print(f"      Out-of-sample Macro F1:      {fc_metrics['test_macro_f1']:.4f}")

    # Backtest
    test_returns = df["log_return"].iloc[-len(y_pred):].values
    bt_results = backtest_regime_strategy(test_returns, y_pred)

    strat = bt_results["strategy"]
    bench = bt_results["benchmark_buy_and_hold"]

    print("\n" + "=" * 72)
    print("  OUT-OF-SAMPLE BACKTEST COMPARISON (TEST HORIZON)")
    print("=" * 72)
    print(f"  {'Metric':<28} | {'Regime Strategy':<18} | {'Buy & Hold':<18}")
    print("  " + "-" * 68)
    print(f"  {'Cumulative Return':<28} | {strat['cumulative_return_pct']:>16}% | {bench['cumulative_return_pct']:>16}%")
    print(f"  {'Annualized CAGR':<28} | {strat['cagr_pct']:>16}% | {bench['cagr_pct']:>16}%")
    print(f"  {'Annualized Volatility':<28} | {strat['annualized_volatility_pct']:>16}% | {bench['annualized_volatility_pct']:>16}%")
    print(f"  {'Sharpe Ratio':<28} | {strat['sharpe_ratio']:>16} | {bench['sharpe_ratio']:>16}")
    print(f"  {'Max Drawdown':<28} | {strat['max_drawdown_pct']:>16}% | {bench['max_drawdown_pct']:>16}%")
    print(f"  {'Win Rate':<28} | {strat['win_rate_pct']:>16}% | {bench['win_rate_pct']:>16}%")
    print("  " + "=" * 68)
    print(f"  ★ Alpha Generated:          +{bt_results['alpha_pct']}%")
    print(f"  ★ Max Drawdown Reduction:   {bt_results['drawdown_reduction_pct']}% lower risk")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Market Regime Detector ML CLI")
    parser.add_argument("--days", type=int, default=1200, help="Number of trading days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_pipeline(n_days=args.days, seed=args.seed)


if __name__ == "__main__":
    main()
