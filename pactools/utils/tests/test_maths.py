from functools import partial

import numpy as np

from pactools.utils.maths import norm, squared_norm, argmax_2d
from pactools.utils.testing import assert_equal
from pactools.utils.testing import assert_raises, assert_array_equal
from pactools.utils.testing import assert_true, assert_array_almost_equal


def test_norm():
    # Test that norm and squared_norm are consistent
    rng = np.random.RandomState(0)
    for sig in (rng.randn(10), rng.randn(4, 3), [1 + 1j, 3, 6], -9):
        assert_array_almost_equal(norm(sig) ** 2, squared_norm(sig))


def test_argmax_2d():
    # Tet that argmax_2d gives the correct indices of the max
    rng = np.random.RandomState(0)
    a = rng.randn(4, 3)
    i, j = argmax_2d(a)
    assert_equal(a[i, j], a.max())
