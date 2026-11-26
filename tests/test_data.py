from market_regime.data import generate_synthetic_market_data


def test_generate_synthetic_market_data():
    df = generate_synthetic_market_data(n_days=200, seed=123)

    assert len(df) == 200
    for col in ["Open", "High", "Low", "Close", "Volume", "TrueRegime"]:
        assert col in df.columns

    # Financial consistency checks
    assert (df["High"] >= df["Low"]).all()
    assert (df["High"] >= df["Open"]).all()
    assert (df["High"] >= df["Close"]).all()
    assert (df["Low"] <= df["Open"]).all()
    assert (df["Low"] <= df["Close"]).all()
    assert (df["Volume"] > 0).all()
