import numpy as np
import pandas as pd
import pytest
from tests.conftest import corwin_schultz_spread


def make_series(*values):
    return pd.Series(values, dtype=float)


def test_known_input_returns_positive_spread():
    # high=110, low=100 for two consecutive days → spread should be positive
    high = make_series(110.0, 110.0, 110.0)
    low = make_series(100.0, 100.0, 100.0)
    result = corwin_schultz_spread(high, low)
    # First two rows are NaN (rolling(2) needs two periods), third should be positive
    assert result.dropna().iloc[-1] > 0


def test_high_equals_low_gives_zero_spread():
    # When high == low, log ratio is 0, spread should be 0
    high = make_series(100.0, 100.0, 100.0)
    low = make_series(100.0, 100.0, 100.0)
    result = corwin_schultz_spread(high, low)
    non_nan = result.dropna()
    assert (non_nan == 0.0).all()


def test_spread_is_non_negative():
    # With arbitrary realistic BIST-like prices, no value should be negative
    rng = np.random.default_rng(0)
    base = 50.0
    closes = base + np.cumsum(rng.normal(0, 1, 30))
    high = pd.Series(closes * rng.uniform(1.00, 1.05, 30))
    low = pd.Series(closes * rng.uniform(0.95, 1.00, 30))
    result = corwin_schultz_spread(high, low)
    assert (result.dropna() >= 0).all()


def test_spread_capped_at_50pct():
    # Extreme H/L ratio should not produce spread > 0.5 (clip upper=0.5)
    high = make_series(200.0, 200.0, 200.0)
    low = make_series(1.0, 1.0, 1.0)
    result = corwin_schultz_spread(high, low)
    assert (result.dropna() <= 0.5).all()
