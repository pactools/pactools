from functools import partial

import numpy as np

from pactools.utils.maths import argmax_2d
from pactools.utils.testing import assert_raises, assert_array_equal
from pactools.utils.testing import assert_true, assert_array_almost_equal
from pactools.comodulogram import comodulogram
from pactools.comodulogram import ALL_PAC_METRICS, BICOHERENCE_PAC_METRICS
from pactools.create_signal import create_signal

low_fq_range = [1., 3., 5., 7.]
high_fq_range = [25., 50., 75.]
n_low = len(low_fq_range)
n_high = len(high_fq_range)
n_points = 1024
fs = 200.

signal = create_signal(
    n_points=n_points, fs=fs, high_fq=high_fq_range[1],
    low_fq=low_fq_range[1], low_fq_width=1., noise_level=0.1, random_state=0)


def fast_comod(**kwargs):
    # A comodulogram call with default params used for testing
    if 'fs' not in kwargs:
        kwargs['fs'] = fs
    if 'low_sig' not in kwargs:
        kwargs['low_sig'] = signal
    if 'low_fq_range' not in kwargs:
        kwargs['low_fq_range'] = low_fq_range
    if 'low_fq_width' not in kwargs:
        kwargs['low_fq_width'] = 1.
    if 'high_fq_range' not in kwargs:
        kwargs['high_fq_range'] = high_fq_range
    if 'high_fq_width' not in kwargs:
        kwargs['high_fq_width'] = 14.
    if 'progress_bar' not in kwargs:
        kwargs['progress_bar'] = False
    if 'random_state' not in kwargs:
        kwargs['random_state'] = 0

    return comodulogram(**kwargs)


def test_input_checking():
    # test that we have a ValueError for bad parameters
    func = partial(fast_comod, method='wrong')
    assert_raises(ValueError, func)
    func = partial(fast_comod, fs='wrong')
    assert_raises(ValueError, func)
    func = partial(fast_comod, low_sig='wrong')
    assert_raises(ValueError, func)
    func = partial(fast_comod, high_sig='wrong')
    assert_raises(ValueError, func)


def test_different_dimension_in_input():
    pass


def test_high_sig_identical():
    # Test that we have the same result with high_sig=low_sig and high_sig=None
    for method in ALL_PAC_METRICS:
        if method in BICOHERENCE_PAC_METRICS:
            continue
        comod_0 = fast_comod(method=method)
        comod_1 = fast_comod(high_sig=signal, method=method)
        assert_array_equal(comod_0, comod_1)


def test_comod_maximum():
    # Test that the PAC is maximum at the correct location in the comodulogram
    for method in ALL_PAC_METRICS:
        comod = fast_comod(method=method)
        # test the shape of the comodulogram
        assert_array_equal(comod.shape, (n_low, n_high))

        # the bicoherence metrics fail this test with current parameters
        if method in BICOHERENCE_PAC_METRICS or method == 'jiang':
            continue
        assert_array_equal(argmax_2d(comod), (1, 1))
        assert_true(np.all(comod > 0))


def test_empty_mask():
    # Test that using an empty mask does not change the results
    mask = np.zeros(n_points, dtype=bool)

    for method in ALL_PAC_METRICS:
        comod_0 = fast_comod(mask=mask, method=method)
        comod_1 = fast_comod(low_sig=signal[~mask], method=method)
        assert_array_almost_equal(comod_0, comod_1, decimal=7)
