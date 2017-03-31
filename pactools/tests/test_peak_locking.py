import numpy as np
import matplotlib.pyplot as plt

from pactools.utils.testing import assert_equal, assert_array_almost_equal
from pactools.utils.testing import assert_raises
from pactools.peak_locking import PeakLocking
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


def fast_peak_locking(low_sig=signal, high_sig=None, mask=None, fs=fs,
                      low_fq=low_fq, *args, **kwargs):
    return PeakLocking(fs=fs, low_fq=low_fq, *args, **kwargs).fit(
        low_sig=low_sig, high_sig=high_sig, mask=mask)


def test_peak_locking_shape():
    percentiles = [5, 25, 50, 75, 95]
    plkg = PeakLocking(fs=fs, low_fq=low_fq, high_fq_range=high_fq_range,
                       percentiles=percentiles)
    plkg.fit(signal)
    n_high, n_window = plkg.time_frequency_.shape
    n_percentiles, n_window_2 = plkg.time_average_.shape
    assert_equal(n_high, len(high_fq_range))
    assert_equal(n_percentiles, len(percentiles))
    assert_equal(n_window, n_window_2)


def test_different_dimension_in_input():
    # Test that 1D or 2D signals are accepted, but not 3D
    for dim in [(4, -1), (-1, ), (1, -1)]:
        fast_peak_locking(signal.reshape(*dim))

    dim = (2, 2, -1)
    assert_raises(ValueError, fast_peak_locking, signal.reshape(*dim))


def test_empty_mask():
    # Test that using an empty mask does not change the results
    mask = np.zeros(n_points, dtype=bool)

    plkg_0 = fast_peak_locking(mask=mask)
    plkg_1 = fast_peak_locking(low_sig=signal[~mask])
    assert_array_almost_equal(plkg_0.time_frequency_, plkg_1.time_frequency_)
    assert_array_almost_equal(plkg_0.time_average_, plkg_1.time_average_)


def test_plot_peaklocking():
    # Smoke test with the standard plotting function
    est = PeakLocking(fs=fs, low_fq=low_fq).fit(signal)
    est.plot()
    plt.close('all')
