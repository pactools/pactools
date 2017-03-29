import numpy as np
from scipy.signal import hilbert

from .utils.spectrum import compute_n_fft
from .utils.carrier import Carrier


def multiple_band_pass(sigs, fs, frequency_range, bandwidth, n_cycles=None,
                       filter_method='carrier'):
    """
    Band-pass filter the signal at multiple frequencies

    Parameters
    ----------
    sigs : array, shape (n_epochs, n_points)
        Input array to filter

    fs : float
        Sampling frequency

    frequency_range : float, list, or array, shape (n_frequencies, )
        List of center frequency of bandpass filters.

    bandwidth : float
        Bandwidth of the bandpass filters.
        Use it to have a constant bandwidth for all filters.
        Should be None if n_cycles is not None.

    n_cycles : float
        Number of cycles of the bandpass filters.
        Use it to have a bandwidth proportional to the center frequency.
        Should be None if bandwidth is not None.

    filter_method : string, in {'mne', 'carrier'}
        Method to bandpass filter.
        - 'carrier' uses internal wavelet-based bandpass filter. (default)
        - 'mne' uses mne.filter.band_pass_filter in MNE-python package

    Returns
    -------
    filtered : array, shape (n_frequencies, n_epochs, n_points)
        Bandpass filtered version of the input signals
    """
    fixed_n_cycles = n_cycles

    sigs = np.atleast_2d(sigs)
    n_fft = compute_n_fft(sigs)
    n_epochs, n_points = sigs.shape

    frequency_range = np.atleast_1d(frequency_range)
    n_frequencies = frequency_range.shape[0]

    if filter_method == 'carrier':
        fir = Carrier()

    filtered = np.zeros((n_frequencies, n_epochs, n_points),
                        dtype=np.complex128)
    for ii in range(n_epochs):
        for jj, frequency in enumerate(frequency_range):
            # evaluate the number of cycle for this bandwidth and frequency
            if fixed_n_cycles is None:
                n_cycles = 1.65 * frequency / bandwidth

            # --------- with mne.filter.band_pass_filter
            if filter_method == 'mne':
                from mne.filter import band_pass_filter
                low_sig = band_pass_filter(
                    sigs[ii, :], Fs=fs, Fp1=frequency - bandwidth / 2.0,
                    Fp2=frequency + bandwidth / 2.0,
                    l_trans_bandwidth=bandwidth / 4.0,
                    h_trans_bandwidth=bandwidth / 4.0, n_jobs=1, method='iir')

            # --------- with pactools.Carrier
            if filter_method == 'carrier':
                fir.design(fs, frequency, n_cycles, None, zero_mean=True)
                low_sig = fir.direct(sigs[ii, :])

            # common to the two methods
            filtered[jj, ii, :] = hilbert(low_sig, n_fft)[:n_points]

    return filtered
