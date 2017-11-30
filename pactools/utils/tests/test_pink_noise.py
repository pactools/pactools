import numpy as np

from pactools.utils.pink_noise import pink_noise, almost_pink_noise
from pactools.utils.spectrum import Spectrum
from pactools.utils.testing import assert_almost_equal


def test_pink_noise_shape():
    n_points = 999

    # test the shape
    for func in almost_pink_noise, pink_noise:
        noise = func(n_points, slope=1.)
        assert noise.shape == (n_points, )


def test_pink_noise_slope():
    n_points = 10000
    fs = 500.0
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        return True

    # test the slope
    for slope in [1, 1.5, 2]:
        noise = pink_noise(n_points, slope=slope)
        spec = Spectrum(fs=fs)
        psd = spec.periodogram(noise).T

        freq = np.linspace(0, fs / 2., psd.size)[:, None]

        # linear regression fit in the log domain
        reg = LinearRegression()
        reg.fit(np.log10(freq[1:]), np.log10(psd[1:]))
        assert_almost_equal(reg.coef_[0][0], -slope, decimal=1)
