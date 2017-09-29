import numpy as np

from pactools.utils.parallel import _FakeParallel, _fake_delayed
from pactools.utils.testing import assert_array_equal, assert_equal


def twice(a):
    return 2 * a


def test_fake_parallel():
    # Test that _FakeParallel does nothing but unrolling the generator
    generator = (twice(b) for b in range(10))
    results = _FakeParallel(n_jobs=1)(generator)
    reference = [twice(b) for b in range(10)]
    assert_array_equal(np.array(results), np.array(reference))

    # test fake parameters
    _FakeParallel(n_jobs=1, foo='bar', baz=42)


def test_fake_delayed():
    # Test that fake_delayed does nothing but returning the input arg
    assert_equal(twice, _fake_delayed(twice))
    assert_equal(42, _fake_delayed(42))
