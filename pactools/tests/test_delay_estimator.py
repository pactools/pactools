import numpy as np
import matplotlib.pyplot as plt

from pactools.utils.testing import assert_equal, assert_array_almost_equal
from pactools.utils.testing import assert_raises, assert_greater
from pactools.delay_estimator import DelayEstimator
from pactools.dar_model import DAR
from pactools.simulate_pac import simulate_pac

# Parameters used for the simulated signal in the test
high_fq = 80.
low_fq = 3.
low_fq_width = 2.
n_points = 1024
fs = 500.

signal = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                      low_fq_width=1., noise_level=0.1, random_state=0)


def fast_delay(low_sig=signal, high_sig=None, mask=None, fs=fs, low_fq=low_fq,
               *args, **kwargs):
    dar_model = DAR(ordar=10, ordriv=2)
    return DelayEstimator(fs=fs, dar_model=dar_model, low_fq=low_fq,
                          low_fq_width=low_fq_width, random_state=0,
                          max_delay=0.01 / low_fq, *args, **kwargs).fit(
                              low_sig=low_sig, high_sig=high_sig, mask=mask)


def test_delay_shape():
    est = fast_delay()
    assert_equal(est.neg_log_likelihood_.shape, est.delays_ms_.shape)
    assert_greater(est.neg_log_likelihood_.shape[0], 1)
    i_best = np.nanargmin(est.neg_log_likelihood_)
    assert_equal(est.best_delay_ms_, est.delays_ms_[i_best])


def test_different_dimension_in_input():
    # Test that 1D or 2D signals are accepted, but not 3D
    for dim in [(4, -1), (-1, ), (1, -1)]:
        fast_delay(signal.reshape(*dim))

    dim = (2, 2, -1)
    assert_raises(ValueError, fast_delay, signal.reshape(*dim))


def test_empty_mask():
    # Test that using an empty mask does not change the results
    mask = np.zeros(n_points, dtype=bool)

    est_0 = fast_delay(mask=mask)
    est_1 = fast_delay()
    assert_array_almost_equal(est_0.neg_log_likelihood_,
                              est_1.neg_log_likelihood_)


def test_plot_delay_estimation():
    # Smoke test with the standard plotting function
    est = fast_delay()
    est.plot()
    plt.close('all')
