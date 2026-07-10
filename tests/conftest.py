import numpy as np
import pandas as pd

K_CS = 3 - 2 * np.sqrt(2)


def corwin_schultz_spread(high, low):
    """Corwin-Schultz (2012) spread estimator. Matches liquidity_premium_test.py."""
    high = high.replace(0, np.nan)
    low = low.replace(0, np.nan)
    hl1 = np.log(high / low) ** 2
    high2 = high.rolling(2).max()
    low2 = low.rolling(2).min()
    hl2 = np.log(high2 / low2) ** 2
    beta = hl1 + hl1.shift(1)
    gamma = hl2
    sqrt_beta = np.sqrt(beta.clip(lower=0))
    sqrt_gamma = np.sqrt(gamma.clip(lower=0))
    alpha = (np.sqrt(2) - 1) * sqrt_beta / K_CS - sqrt_gamma / np.sqrt(K_CS)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_a = np.exp(alpha)
        spread = 2 * (exp_a - 1) / (1 + exp_a)
    return spread.clip(lower=0, upper=0.5)


def bh_fdr(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Matches liquidity_premium_test.py."""
    p = np.array(p_values)
    n = len(p)
    ranks = np.argsort(p) + 1
    q = np.empty(n)
    q[np.argsort(p)] = p * n / ranks
    for i in range(n - 2, -1, -1):
        q[i] = min(q[i], q[i + 1])
    return q.clip(max=1.0)
