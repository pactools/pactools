import numpy as np

from pactools.utils.maths import norm, squared_norm, argmax_2d, is_power2
from pactools.utils.maths import prime_factors
from pactools.utils.testing import assert_equal
from pactools.utils.testing import assert_array_almost_equal
from pactools.utils.testing import assert_true, assert_false


def test_norm():
    # Test that norm and squared_norm are consistent
    rng = np.random.RandomState(0)
    for sig in (rng.randn(10), rng.randn(4, 3), [1 + 1j, 3, 6], -9):
        assert_array_almost_equal(norm(sig) ** 2, squared_norm(sig))


def test_argmax_2d():
    # Test that argmax_2d gives the correct indices of the max
    rng = np.random.RandomState(0)
    a = rng.randn(4, 3)
    i, j = argmax_2d(a)
    assert_equal(a[i, j], a.max())


def test_is_power2():
    for i in range(1, 10):
        assert_true(is_power2(2 ** i))
        assert_false(is_power2(2 ** i + 1))


def test_prime_factors():
    # Test that the prime factor decomposition products to the input integer
    for n in range(3, 200, 2):
        factors = prime_factors(n)
        assert_equal(np.product(factors), n)
