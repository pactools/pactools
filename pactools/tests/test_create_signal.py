import numpy as np

from pactools.utils.testing import assert_equal, assert_raises
from pactools.utils.testing import assert_array_almost_equal
from pactools.utils.testing import assert_array_equal, assert_almost_equal
from pactools.simulate_pac import simulate_pac

low_fq_range = [1., 3., 5., 7.]
high_fq_range = [25., 50., 75.]
n_low = len(low_fq_range)
n_high = len(high_fq_range)
n_points = 1024


def simulate_pac_default(n_points=128, fs=200., high_fq=50., low_fq=3.,
                         low_fq_width=1., noise_level=0.1, random_state=42,
                         *args, **kwargs):
    """Call simulate_pac with default values, used for testing only."""
    return simulate_pac(n_points, fs, high_fq, low_fq, low_fq_width,
                        noise_level, random_state=random_state, *args,
                        **kwargs)


def test_correct_size():
    # Test that the signals are 1D and with the correct number of points
    for n_points_in in np.int_(np.logspace(1, 3, 5)):
        signal = simulate_pac_default(n_points=n_points_in)
        assert_equal(signal.size, n_points_in)
        assert_equal(signal.ndim, 1)


def test_random_state():
    # Test that the random state controls reproducibility
    sig_0 = simulate_pac_default(random_state=0)
    sig_1 = simulate_pac_default(random_state=1)
    sig_2 = simulate_pac_default(random_state=0)
    assert_raises(AssertionError, assert_array_almost_equal, sig_0, sig_1)
    assert_array_almost_equal(sig_0, sig_2)


def test_amplitude():
    # Test that the amplitude parameters change the amplitude
    sig_0 = simulate_pac_default(high_fq_amp=0, low_fq_amp=0, noise_level=0)
    zeros = np.zeros_like(sig_0)
    assert_array_equal(sig_0, zeros)

    sig_1 = simulate_pac_default(high_fq_amp=1, low_fq_amp=0, noise_level=0)
    sig_2 = simulate_pac_default(high_fq_amp=2, low_fq_amp=0, noise_level=0)
    # with a driver equal to zero at all time, the sigmoid modulates at 0.5
    assert_almost_equal(sig_1.std(), 0.5, decimal=7)
    assert_almost_equal(sig_2.std(), 1, decimal=7)

    sig_1 = simulate_pac_default(high_fq_amp=0, low_fq_amp=0, noise_level=1)
    sig_2 = simulate_pac_default(high_fq_amp=0, low_fq_amp=0, noise_level=2)
    assert_almost_equal(sig_1.std(), 1, decimal=1)
    assert_almost_equal(sig_2.std(), 2, decimal=1)

    sig_1 = simulate_pac_default(high_fq_amp=0, low_fq_amp=1, noise_level=0)
    sig_2 = simulate_pac_default(high_fq_amp=0, low_fq_amp=2, noise_level=0)
    assert_almost_equal(sig_1.std(), 1, decimal=0)
    assert_almost_equal(sig_2.std(), 2, decimal=0)
