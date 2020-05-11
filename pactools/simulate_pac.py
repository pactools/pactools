import numpy as np

from .utils.fir import BandPassFilter
from .utils.validation import check_random_state
from .utils.deprecation import ignore_warnings


@ignore_warnings(category=RuntimeWarning)
def sigmoid(array, sharpness):
    result = 1. / (1. + np.exp(-sharpness * array))
    assert np.all(np.isfinite(result))
    return result


def simulate_pac(n_points, fs, high_fq, low_fq, low_fq_width, noise_level,
                 high_fq_amp=0.5, low_fq_amp=0.5, random_state=None,
                 sigmoid_sharpness=6, phi_0=0., delay=0., return_driver=False,
                 signal_len=None):
    """Simulate a 1D signal with artificial phase amplitude coupling (PAC).

    Parameters
    ----------

    n_points : int
        Number of points in the signal- must be None if signal_len is not None

    fs : float
        Sampling frequency of the signal

    high_fq : float
        Frequency of the fast oscillation which is modulated in amplitude

    low_fq : float
        Center frequency of the slow oscillation which controls the modulation

    low_fq_width : float
        Bandwidth of the slow oscillation which controls the modulation

    noise_level : float
        Level of the Gaussian additive white noise

    high_fq_amp : float
        Amplitude of the fast oscillation

    low_fq_amp : float
        Amplitude of the slow oscillation

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator.

    sigmoid_sharpness : float
        Sharpness of the sigmoid used to define the modulation

    phi_0 : float
        Preferred phase of the coupling: phase of the slow oscillation which
        corresponds to the maximum amplitude of the fast oscillations

    delay : float
        Delay between the slow oscillation and the modulation

    return_driver : boolean
        If True, return the complex driver instead of the full signal

    signal_len : int
        Length of the signal in seconds- cannot be None if n_points is not None

    Returns
    -------
    signal : array, shape (signal_len * fs, )
        Signal with artifical PAC

    """
    if ((n_points is None and signal_len is None
         ) or (n_points is not None and signal_len is not None)):
        raise ValueError("'n_points' xor 'signal_len' must be defined")
    if signal_len is not None:
        n_points = int(signal_len * fs)
    fs = float(fs)
    rng = check_random_state(random_state)
    if high_fq >= fs / 2 or low_fq >= fs / 2:
        raise ValueError('Frequency is larger or equal to half '
                         'the sampling frequency.')
    if high_fq < 0 or low_fq < 0 or fs < 0:
        raise ValueError('Invalid negative frequency')

    fir = BandPassFilter(fs=fs, fc=low_fq, n_cycles=None,
                         bandwidth=low_fq_width, extract_complex=True)
    driver_real, driver_imag = fir.transform(rng.randn(n_points))
    driver = driver_real + 1j * driver_imag
    # We scale by sqrt(2) to have correct amplitude in the real-valued driver
    driver *= 1. / driver.std() * np.sqrt(2)
    if return_driver:
        return driver

    # decay of sigdriv for continuity after np.roll
    if delay != 0:
        n_decay = max(int(0.5 * fs / low_fq), 5)
        window = np.blackman(n_decay * 2 - 1)[:n_decay]
        driver[:n_decay] *= window
        driver[-n_decay:] *= window[::-1]

    # create the fast oscillation
    time = np.arange(n_points) / float(fs)
    carrier = np.sin(2 * np.pi * high_fq * time)
    carrier *= high_fq_amp / carrier.std()

    # create the modulation, with a sigmoid, and a phase lag
    phase = np.exp(1j * phi_0)
    modulation = sigmoid(np.real(driver * phase), sharpness=sigmoid_sharpness)

    # the slow oscillation is not phased lag
    theta = np.real(driver) * low_fq_amp

    # add a delay to the slow oscillation theta
    delay_point = int(delay * fs)
    theta = np.roll(theta, delay_point)

    # apply modulation
    gamma = carrier * modulation

    # create noise
    noise = rng.randn(n_points) * noise_level
    # add all oscillations
    signal = gamma + theta + noise

    return signal
