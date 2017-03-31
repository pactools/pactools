from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from pactools.dar_model import AR, DAR, HAR, StableDAR
from pactools.utils.testing import assert_equal
from pactools.utils.testing import assert_raises, assert_array_equal
from pactools.utils.testing import assert_true, assert_array_almost_equal
from pactools.comodulogram import Comodulogram
from pactools.comodulogram import ALL_PAC_METRICS, BICOHERENCE_PAC_METRICS
from pactools.simulate_pac import simulate_pac

# Parameters used for the simulated signal in the test
low_fq_range = [1., 3., 5., 7.]
high_fq_range = [25., 50., 75.]
n_low = len(low_fq_range)
n_high = len(high_fq_range)
high_fq = high_fq_range[1]
low_fq = low_fq_range[1]
n_points = 1024
fs = 200.

signal = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                      low_fq_width=1., noise_level=0.1, random_state=0)
signal_copy = signal.copy()


class ComodTest(Comodulogram):
    # A comodulogram call with default params used for testing
    def __init__(self, fs=fs, low_fq_range=low_fq_range, low_fq_width=1.,
                 high_fq_range=high_fq_range, high_fq_width='auto',
                 method='tort', n_surrogates=0, vmin=None, vmax=None,
                 progress_bar=False, ax_special=None, minimum_shift=1.0,
                 random_state=0, coherence_params=dict(), low_fq_width_2=4.0):
        super(ComodTest, self).__init__(
            fs=fs, low_fq_range=low_fq_range, low_fq_width=low_fq_width,
            high_fq_range=high_fq_range, high_fq_width=high_fq_width,
            method=method, n_surrogates=n_surrogates, vmin=vmin, vmax=vmax,
            progress_bar=progress_bar, ax_special=ax_special,
            minimum_shift=minimum_shift, random_state=random_state,
            coherence_params=coherence_params, low_fq_width_2=low_fq_width_2)


def fast_comod(low_sig=signal, high_sig=None, mask=None, *args, **kwargs):
    return ComodTest(*args, **kwargs).fit(low_sig=low_sig, high_sig=high_sig,
                                          mask=mask).comod_


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
    # Test that 1D or 2D signals are accepted, but not 3D
    for dim in [
        (4, -1),
        (-1, ),
        (1, -1),
    ]:
        fast_comod(signal.reshape(*dim))

    dim = (2, 2, -1)
    assert_raises(ValueError, fast_comod, signal.reshape(*dim))


def test_high_sig_identical():
    # Test that we have the same result with high_sig=low_sig and high_sig=None
    for method in ALL_PAC_METRICS:
        if method in BICOHERENCE_PAC_METRICS:
            continue
        comod_0 = fast_comod(method=method)
        comod_1 = fast_comod(high_sig=signal, method=method)
        assert_array_equal(comod_0, comod_1)


def test_comod_correct_maximum():
    # Test that the PAC is maximum at the correct location in the comodulogram
    for method in ALL_PAC_METRICS:
        est = ComodTest(method=method, progress_bar=True).fit(signal)
        comod = est.comod_
        # test the shape of the comodulogram
        assert_array_equal(comod.shape, (n_low, n_high))

        # the bicoherence metrics fail this test with current parameters
        if method in BICOHERENCE_PAC_METRICS or method == 'jiang':
            continue

        low_fq_0, high_fq_0, max_pac = est.get_maximum_pac()
        assert_equal(low_fq_0, low_fq)
        assert_equal(high_fq_0, high_fq)
        assert_equal(max_pac, comod.max())
        assert_true(np.all(comod > 0))


def test_empty_mask():
    # Test that using an empty mask does not change the results
    mask = np.zeros(n_points, dtype=bool)

    for method in ALL_PAC_METRICS:
        comod_0 = fast_comod(mask=mask, method=method)
        comod_1 = fast_comod(low_sig=signal[~mask], method=method)
        assert_array_almost_equal(comod_0, comod_1, decimal=7)


def test_comodulogram_dar_models():
    # Smoke test with DAR models
    for klass in (AR, DAR, HAR, StableDAR):
        if klass is StableDAR:
            model = klass(ordar=10, ordriv=2, iter_newton=10)
        else:
            model = klass(ordar=10, ordriv=2)
        comod = fast_comod(method=model)
        assert_true(~np.any(np.isnan(comod)))


def test_plot_comodulogram():
    # Smoke test with the standard plotting function
    est = ComodTest().fit(signal)
    est.plot()

    # Smoke test with the special plotting functions
    ax = plt.figure().gca()
    for method in ALL_PAC_METRICS:
        est = ComodTest(low_fq_range=[low_fq], method=method,
                        ax_special=ax).fit(signal)

    # Test that it raises an error if ax_special is not None and low_fq_range
    # has more than one element
    func = partial(fast_comod, ax_special=ax)
    assert_raises(ValueError, func)
    plt.close('all')


def test_signal_unchanged():
    # Test that signal has not been changed during the test
    assert_array_equal(signal_copy, signal)
