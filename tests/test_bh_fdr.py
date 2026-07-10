import numpy as np
import pytest
from tests.conftest import bh_fdr


def test_known_p_values():
    # Hand-computed BH-FDR for p=[0.01, 0.04, 0.20], n=3
    # Sorted ranks: p=0.01→rank1, p=0.04→rank2, p=0.20→rank3
    # Raw q: 0.01*3/1=0.03, 0.04*3/2=0.06, 0.20*3/3=0.20
    # Monotone: 0.03, 0.06, 0.20  (already monotone)
    p = [0.01, 0.04, 0.20]
    q = bh_fdr(p)
    expected = np.array([0.03, 0.06, 0.20])
    np.testing.assert_allclose(q, expected, atol=1e-10)


def test_all_ones_return_one():
    p = [1.0, 1.0, 1.0]
    q = bh_fdr(p)
    assert (np.array(q) == 1.0).all()


def test_q_values_bounded():
    # q-values must be in [0, 1]
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50).tolist()
    q = bh_fdr(p)
    assert (np.array(q) >= 0).all()
    assert (np.array(q) <= 1).all()
