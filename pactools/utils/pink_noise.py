import numpy as np

from .validation import check_random_state


def almost_pink_noise(n_points, slope=1., plateau=0.05, random_state=None):
    """Generate some pink (1/f^slope) noise, with a plateau at low frequencies

    Parameters
    ----------
    n_points : integer
        Length of the noise signal

    slope : float
        The generated noise will have a (1/f^slope) energy

    plateau : float
        The frequencies will have the same energy between 0 and plateau
    """
    n_points = int(n_points)
    rng = check_random_state(random_state)
    uneven = n_points % 2
    x = (rng.randn(n_points // 2 + 1 + uneven) +
         rng.randn(n_points // 2 + 1 + uneven) * 1j)

    scaling = np.sqrt(np.arange(1, x.size + 1) ** slope)
    frequencies = np.linspace(0.0, 1.0, x.size)
    scaling[frequencies < plateau] = scaling[frequencies >= plateau][0]

    x /= scaling
    y = np.fft.irfft(x).real
    if uneven:
        y = y[:-1]
    return y / y.std()


def pink_noise(n_points, slope=1., random_state=None):
    """Generate some pink (1/f^slope) noise

    Parameters
    ----------
    n_points : integer
        Length of the noise signal

    slope : float
        The generated noise will have a (1/f^slope) energy
    """
    return almost_pink_noise(n_points=n_points, slope=slope, plateau=0.,
                             random_state=random_state)
