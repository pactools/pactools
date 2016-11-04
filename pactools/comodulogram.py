import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.interpolate import interp1d, interp2d

from .dar_model import DAR, SimpleDAR
from .utils.progress_bar import ProgressBar
from .utils.spectrum import compute_n_fft, Bicoherence, Coherence
from .utils.carrier import Carrier
from .utils.maths import norm, argmax_2d, check_random_state
from .plot_comodulogram import plot_comodulogram
from .preprocess import extract


def multiple_band_pass(sigs, fs, frequency_range, bandwidth,
                       n_cycles=None, filter_method='carrier'):
    """
    Band-pass filter the signal at multiple frequencies
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
                    sigs[ii, :], Fs=fs,
                    Fp1=frequency - bandwidth / 2.0,
                    Fp2=frequency + bandwidth / 2.0,
                    l_trans_bandwidth=bandwidth / 4.0,
                    h_trans_bandwidth=bandwidth / 4.0,
                    n_jobs=1, method='iir')

            # --------- with pactools.Carrier
            if filter_method == 'carrier':
                fir.design(fs, frequency, n_cycles, None, zero_mean=True)
                low_sig = fir.direct(sigs[ii, :])

            # common to the two methods
            filtered[jj, ii, :] = hilbert(low_sig, n_fft)[:n_points]

    return filtered


def _comodulogram(fs, filtered_low, filtered_high, mask, method, n_surrogates,
                  progress_bar, draw_phase, minimum_shift, random_state,
                  filtered_low_2):
    """
    Compute the comodulogram for empirical metrics.
    """
    # The modulation index is only computed where mask is True
    if mask is not None:
        filtered_low = filtered_low[:, mask == 1]
        filtered_high = filtered_high[:, mask == 1]
        if method == 'vanwijk':
            filtered_low_2 = filtered_low_2[:, mask == 1]
    else:
        filtered_low = filtered_low.reshape(filtered_low.shape[0], -1)
        filtered_high = filtered_high.reshape(filtered_high.shape[0], -1)
        if method == 'vanwijk':
            filtered_low_2 = filtered_low_2.reshape(filtered_low_2.shape[0],
                                                    -1)

    n_low, n_points = filtered_low.shape
    n_high, _ = filtered_high.shape

    # phase of the low frequency signals
    for i in range(n_low):
        filtered_low[i] = np.angle(filtered_low[i])
    filtered_low = np.real(filtered_low)

    # amplitude of the high frequency signals
    filtered_high = np.real(np.abs(filtered_high))
    norm_a = np.zeros(n_high)
    if method == 'ozkurt':
        for j in range(n_high):
            norm_a[j] = norm(filtered_high[j])

    # amplitude of the low frequency signals
    if method == 'vanwijk':
        for i in range(n_low):
            filtered_low_2[i] = np.abs(filtered_low_2[i])
        filtered_low_2 = np.real(filtered_low_2)

    # Calculate the modulation index for each couple
    comod = np.zeros((n_low, n_high))
    for i in range(n_low):
        # preproces the phase array
        if method == 'tort':
            n_bins = 18
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            # get the indices of the bins to which each value in input belongs
            phase_preprocessed = np.digitize(filtered_low[i], phase_bins) - 1
        elif method == 'penny':
            phase_preprocessed = np.c_[np.ones_like(filtered_low[i]),
                                       np.cos(filtered_low[i]),
                                       np.sin(filtered_low[i])]
        elif method == 'vanwijk':
            phase_preprocessed = np.c_[np.ones_like(filtered_low[i]),
                                       np.cos(filtered_low[i]),
                                       np.sin(filtered_low[i]),
                                       filtered_low_2[i]]
        else:
            phase_preprocessed = np.exp(1j * filtered_low[i])

        for j in range(n_high):

            def comod_function(shift):
                return _one_modulation_index(
                    amplitude=filtered_high[j],
                    phase_preprocessed=phase_preprocessed,
                    norm_a=norm_a[j], method=method,
                    shift=shift, draw_phase=draw_phase)

            comod[i, j] = _surrogate_analysis(comod_function, fs, n_points,
                                              minimum_shift, random_state,
                                              n_surrogates)

        if progress_bar:
            progress_bar.update_with_increment_value(1)

    return comod


def _one_modulation_index(amplitude, phase_preprocessed, norm_a, method,
                          shift, draw_phase):
    # shift for the surrogate analysis
    if shift != 0:
        phase_preprocessed = np.roll(phase_preprocessed, shift)

    # Modulation index as in [Ozkurt & al 2011]
    if method == 'ozkurt':
        MI = np.abs(np.mean(amplitude * phase_preprocessed))
        MI *= np.sqrt(amplitude.size) / norm_a

    # Modulation index as in [Penny & al 2008] or [van Wijk & al 2015]
    elif method in ('penny', 'vanwijk', ):
        # solve a linear regression problem:
        # amplitude = np.dot(phase_preprocessed) * beta
        PtP = np.dot(phase_preprocessed.T, phase_preprocessed)
        PtA = np.dot(phase_preprocessed.T, amplitude[:, None])
        beta = np.linalg.solve(PtP, PtA)
        residual = amplitude - np.dot(phase_preprocessed, beta).ravel()
        variance_amplitude = np.var(amplitude)
        variance_residual = np.var(residual)
        MI = (variance_amplitude - variance_residual) / variance_amplitude

    # Modulation index as in [Canolty & al 2006]
    elif method == 'canolty':
        MI = np.abs(np.mean(amplitude * phase_preprocessed))

    # Modulation index as in [Tort & al 2010]
    elif method == 'tort':
        # mean amplitude distribution along phase bins
        for n_bins in range(18, 8, -2):
            amplitude_dist = np.zeros(n_bins)
            for b in range(n_bins):
                selection = amplitude[phase_preprocessed == b]
                if selection.size == 0:  # no sample in that bin
                    continue
                amplitude_dist[b] = np.mean(selection)
            if np.any(amplitude_dist == 0):
                continue
            break

        if np.any(amplitude_dist == 0):
            raise RuntimeError("Not enough data to fill %d bins !" % n_bins)

        # Kullback-Leibler divergence of the distribution vs uniform
        amplitude_dist /= np.sum(amplitude_dist)
        divergence_kl = np.sum(amplitude_dist *
                               np.log(amplitude_dist * n_bins))

        MI = divergence_kl / np.log(n_bins)

        if draw_phase and shift == 0:
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            phase_bins = 0.5 * (phase_bins[:-1] + phase_bins[1:]) / np.pi * 180
            plt.plot(phase_bins, amplitude_dist, '.-')
            plt.plot(phase_bins, np.ones(n_bins) / n_bins, '--')
            plt.ylim((0, 2. / n_bins))
            plt.xlim((-180, 180))
            plt.ylabel('Normalized mean amplitude')
            plt.xlabel('Phase (in degree)')
            plt.title('Tort index: %.3f' % MI)

    else:
        raise ValueError("Unknown method: %s" % (method, ))

    return MI


def _same_mask_on_all_epochs(sig, mask, method):
    mask = np.squeeze(mask)
    if mask.ndim > 1:
        warnings.warn("For coherence methods (e.g. %s) the mask has "
                      "to be unidimensional, and the same mask is "
                      "applied on all epochs. Got shape %s, so only the "
                      "first row of the mask is used." %
                      (method, mask.shape, ), UserWarning)
        mask = mask[0, :]
    sig = sig[..., mask == 1]
    return sig


def _bicoherence(fs, sig, mask, method, low_fq_range,
                 low_fq_width, high_fq_range, coherence_params):
    """Compute the PAC with the bicoherence."""
    # The modulation index is only computed where mask is True
    if mask is not None:
        sig = _same_mask_on_all_epochs(sig, mask, method)

    n_epochs, n_points = sig.shape

    coherence_params = _define_default_params(fs, low_fq_width, True,
                                              **coherence_params)

    estimator = Bicoherence(**coherence_params)
    bicoh = estimator.fit(sigs=sig, method=method)

    # remove the redundant part
    n_freq = bicoh.shape[0]
    np.flipud(bicoh)[np.triu_indices(n_freq, 1)] = 0
    bicoh[np.triu_indices(n_freq, 1)] = 0

    frequencies = np.linspace(0, fs / 2., n_freq)
    comod = _interpolate(frequencies, frequencies, bicoh,
                         high_fq_range, low_fq_range)

    return comod


def _define_default_params(fs, low_fq_width, is_bicoherence, **user_params):
    """Define default values for Coherence and Bicoherence classes,
    if not defined in user_params dictionary."""

    # the FFT length is chosen to have a frequency resolution of low_fq_width
    fft_length = fs / low_fq_width
    # but it is faster if it is a power of 2
    fft_length = 2 ** int(np.ceil(np.log2(fft_length)))

    # smoothing for bicoherence methods
    if is_bicoherence:
        fft_length /= 4

    # the block length is chosen to avoid zero-padding
    block_length = fft_length

    if 'block_length' not in user_params and 'fft_length' not in user_params:
        user_params['block_length'] = block_length
        user_params['fft_length'] = fft_length
    elif 'block_length' in user_params and 'fft_length' not in user_params:
        user_params['fft_length'] = user_params['block_length']
    elif 'block_length' not in user_params and 'fft_length' in user_params:
        user_params['block_length'] = user_params['fft_length']

    if 'fs' not in user_params:
        user_params['fs'] = fs
    if 'step' not in user_params:
        user_params['step'] = None

    return user_params


def _coherence(fs, low_sig, filtered_high, mask, method, low_fq_range,
               low_fq_width, n_surrogates, minimum_shift, random_state,
               progress_bar, coherence_params):
    """Compute the PAC with the coherence."""
    if mask is not None:
        low_sig = _same_mask_on_all_epochs(low_sig, mask, method)
        filtered_high = _same_mask_on_all_epochs(
            filtered_high, mask, method)

    # amplitude of the high frequency signals
    filtered_high = np.real(np.abs(filtered_high))

    coherence_params = _define_default_params(fs, low_fq_width, False,
                                              **coherence_params)

    n_epochs, n_points = low_sig.shape

    def comod_function(shift):
        return _one_coherence_modulation_index(
            fs, low_sig, filtered_high, method, low_fq_range, coherence_params,
            shift)

    comod = _surrogate_analysis(comod_function, fs, n_points, minimum_shift,
                                random_state, n_surrogates)

    return comod


def _one_coherence_modulation_index(fs, low_sig, filtered_high, method,
                                    low_fq_range, coherence_params, shift):
    if shift != 0:
        low_sig = np.roll(low_sig, shift)

    # the actual frequency resolution is computed here
    delta_freq = fs / coherence_params['fft_length']

    estimator = Coherence(**coherence_params)
    coherence = estimator.fit(low_sig[None, :, :], filtered_high)[0]
    n_high, n_freq = coherence.shape
    frequencies = np.linspace(0, fs / 2., n_freq)

    # Coherence as in [Colgin & al 2009]
    if method == 'colgin':
        coherence = np.real(np.abs(coherence))

        comod = _interpolate(np.arange(n_high), frequencies, coherence,
                             np.arange(n_high), low_fq_range)

    # Phase slope index as in [Jiang & al 2015]
    elif method == 'jiang':
        product = coherence[:, 1:] * np.conjugate(coherence[:, :-1])

        # we use a kernel of (ker * 2) with respect to the product,
        # i.e. a kernel of (ker * 2 + 1) with respect to the coherence.
        ker = 2
        kernel = np.ones(2 * ker) / (2 * ker)
        phase_slope_index = np.zeros((n_high, n_freq - (2 * ker)),
                                     dtype=np.complex128)
        for i in range(n_high):
            phase_slope_index[i] = np.convolve(product[i], kernel, 'valid')
        phase_slope_index = np.imag(phase_slope_index)
        frequencies = frequencies[ker:-ker]

        # transform the phase slope index into an approximated delay
        delay = phase_slope_index / (2. * np.pi * delta_freq)

        comod = _interpolate(np.arange(n_high), frequencies, delay,
                             np.arange(n_high), low_fq_range)

    else:
        raise ValueError('Unknown method %s' % (method, ))

    return comod


def _interpolate(x1, y1, z1, x2, y2):
    """Helper to interpolate in 1d or 2d

    We interpolate to get the same shape than with other methods.
    """
    if x1.size > 1 and y1.size > 1:
        func = interp2d(x1, y1, z1.T, kind='linear', bounds_error=False)
        z2 = func(x2, y2)
    elif x1.size == 1 and y1.size > 1:
        func = interp1d(y1, z1.ravel(), kind='linear', bounds_error=False)
        z2 = func(y2)
    elif y1.size == 1 and x1.size > 1:
        func = interp1d(x1, z1.ravel(), kind='linear', bounds_error=False)
        z2 = func(x2)
    else:
        raise ValueError("You can't interpolate a scalar.")

    # interp2d is not intuitive and return this shape:
    z2.shape = (y2.size, x2.size)
    return z2


def comodulogram(fs, low_sig, high_sig=None, mask=None,
                 low_fq_range=np.linspace(1.0, 10.0, 50),
                 high_fq_range=np.linspace(5.0, 150.0, 60),
                 low_fq_width=0.5,
                 high_fq_width=10.0,
                 method='tort',
                 n_surrogates=0,
                 draw=False,
                 vmin=None, vmax=None,
                 progress_bar=True,
                 draw_phase=False,
                 minimum_shift=1.0,
                 random_state=None,
                 coherence_params=dict(),
                 low_fq_width_2=4.0):
    """
    Compute the comodulogram for Phase Amplitude Coupling (PAC).

    Parameters
    ----------
    fs : float,
        Sampling frequency

    low_sig : array, shape (n_epochs, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (n_epochs, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals

    mask : array or list of array or None, shape (n_epochs, n_points)
        The PAC is only evaluated with the unmasked element of low_sig and
        high_sig. Masking is done after filtering and Hilbert transform.
        If the method computes the bicoherence, the mask has to be
        unidimensional (n_points, ) and the same mask is applied on all epochs.
        If a list is given, the filtering is done only once and the
        comodulogram is computed on each mask.

    low_fq_range : array or list
        List of filtering frequencies (phase signal)

    high_fq_range : array or list
        List of filtering frequencies (amplitude signal)

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    high_fq_width : float
        Bandwidth of the band-pass filter (amplitude signal)

    method : string or DAR instance
        Modulation index method:
            - String in ('ozkurt', 'canolty', 'tort', 'penny', ), for a PAC
                estimation based on filtering and using the Hilbert transform.
            - String in ('vanwijk', ) for a joint AAC and PAC estimation
                based on filtering and using the Hilbert transform.
            - String in ('sigl', 'nagashima', 'hagihira', 'bispectrum', ), for
                a PAC estimation based on the bicoherence.
            - String in ('colgin', ) for a PAC estimation
                and in ('jiang', ) for a PAC directionality estimation,
                based on filtering and computing coherence.
            - DAR instance, for a PAC estimation based on a driven
                autoregressive model.

    n_surrogates : int
        Number of surrogates computed for the z-score

    draw : boolean
        If True, plot the comodulogram

    vmin, vmax : float or None
        If not None, it define the min/max value of the plot

    progress_bar : boolean
        If True, a progress bar is shown in stdout

    draw_phase : boolean
        If True, plot the phase distribution in 'tort' index

    minimum_shift : float
        Minimum time shift (in sec) for the surrogate analysis

    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the surrogate analysis

    coherence_params : dict
        Parameters for methods base on coherence or bicoherence.
        May contain:
            block_length : int
                Block length
            fft_length : int or None
                Length of the FFT
            step : int or None
                Step between two blocks
        If the dictionary is empty, default values will be applied based on
        fs and low_fq_width, with 0.5 overlap windows and no zero-padding.

    low_fq_width_2 : float
        Bandwidth of the band-pass filters centered on low_fq_range, for
        the amplitude signal. Used only with 'vanwijk' method.

    Return
    ------
    comod : array, shape (len(low_fq_range), len(high_fq_range))
        Comodulogram for each couple of frequencies.
        If a list of mask is given, it returns a list of comodulograms.
    """
    random_state = check_random_state(random_state)
    if isinstance(method, str):
        method = method.lower()

    # convert to numpy array
    low_fq_range = np.asarray(low_fq_range)
    high_fq_range = np.asarray(high_fq_range)

    multiple_masks = (isinstance(mask, list) or
                      (isinstance(mask, np.ndarray) and mask.ndim == 3))
    if not multiple_masks:
        mask = [mask]
    n_masks = len(mask)

    if method in ('ozkurt', 'canolty', 'tort', 'penny', 'vanwijk'):
        if high_sig is None:
            high_sig = low_sig

        if progress_bar:
            progress_bar = ProgressBar('comodulogram: %s' % method,
                                       max_value=low_fq_range.size * n_masks)

        # compute a number of band-pass filtered and Hilbert filtered signals
        filtered_high = multiple_band_pass(high_sig, fs,
                                           high_fq_range, high_fq_width)
        filtered_low = multiple_band_pass(low_sig, fs,
                                          low_fq_range, low_fq_width)
        if method == 'vanwijk':
            filtered_low_2 = multiple_band_pass(low_sig, fs,
                                                low_fq_range, low_fq_width_2)
        else:
            filtered_low_2 = None

        comod_list = []
        for this_mask in mask:
            comod = _comodulogram(fs, filtered_low, filtered_high, this_mask,
                                  method, n_surrogates, progress_bar,
                                  draw_phase, minimum_shift, random_state,
                                  filtered_low_2)
            comod_list.append(comod)

    elif method in ('jiang', 'colgin'):
        if high_sig is None:
            high_sig = low_sig

        if progress_bar:
            progress_bar = ProgressBar('coherence: %s' % method,
                                       max_value=n_masks)

        # compute a number of band-pass filtered and Hilbert filtered signals
        filtered_high = multiple_band_pass(high_sig, fs,
                                           high_fq_range, high_fq_width)

        comod_list = []
        for this_mask in mask:
            comod = _coherence(fs, low_sig, filtered_high, this_mask, method,
                               low_fq_range, low_fq_width, n_surrogates,
                               minimum_shift, random_state, progress_bar,
                               coherence_params)
            comod_list.append(comod)
            if progress_bar:
                progress_bar.update_with_increment_value(1)

    # compute PAC with the bispectrum/bicoherence
    elif method in ('sigl', 'nagashima', 'hagihira', 'bispectrum'):
        if high_sig is not None:
            raise ValueError(
                "Impossible to use a bicoherence method (%s) on two signals, "
                "please try another method." % method)
        if n_surrogates > 1:
            raise NotImplementedError(
                "Surrogate analysis with a bicoherence method (%s) "
                "is not implemented." % method)

        if progress_bar:
            progress_bar = ProgressBar('bicoherence: %s' % method,
                                       max_value=n_masks)

        comod_list = []
        for this_mask in mask:
            comod = _bicoherence(fs=fs, sig=low_sig,
                                 mask=this_mask, method=method,
                                 low_fq_range=low_fq_range,
                                 low_fq_width=low_fq_width,
                                 high_fq_range=high_fq_range,
                                 coherence_params=coherence_params)
            comod_list.append(comod)
            if progress_bar:
                progress_bar.update_with_increment_value(1)

    elif isinstance(method, (SimpleDAR, DAR)):
        comod_list = driven_comodulogram(fs=fs, low_sig=low_sig,
                                         high_sig=high_sig,
                                         mask=mask, model=method,
                                         low_fq_range=low_fq_range,
                                         low_fq_width=low_fq_width,
                                         high_fq_range=high_fq_range,
                                         progress_bar=progress_bar,
                                         n_surrogates=n_surrogates,
                                         random_state=random_state,
                                         minimum_shift=minimum_shift)
    else:
        raise ValueError('unknown method: %s' % method)

    if draw:
        contours = 4.0 if n_surrogates > 1 else None
        plot_comodulogram(comod_list, fs, low_fq_range, high_fq_range,
                          contours=contours)

    if not multiple_masks:
        return comod_list[0]
    else:
        return comod_list


def driven_comodulogram(fs, low_sig, high_sig, mask, model, low_fq_range,
                        high_fq_range, low_fq_width, method='firstlast',
                        fill=0, ordar=12, enf=50., random_noise=None,
                        normalize=True, whitening='after',
                        progress_bar=True, n_surrogates=0, random_state=None,
                        minimum_shift=1.0):
    """
    Compute the comodulogram with a DAR model.

    Parameters
    ----------
    fs : float,
        Sampling frequency

    low_sig : array, shape (n_epochs, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (n_epochs, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals

    mask : array or list of array or None, shape (n_epochs, n_points)
        The PAC is only evaluated with the unmasked element of low_sig and
        high_sig. Masking is done after filtering and Hilbert transform.
        If a list is given, the filtering is done only once and the
        comodulogram is computed on each mask.

    model : DAR instance
        DAR model to be used for the comodulogram

    low_fq_range : array or list
        List of filtering frequencies (phase signal)

    high_fq_range : array or list
        List of filtering frequencies (amplitude signal). This is not used for
        filtering since DAR models do not need filtering of high frequencies.
        This is only used to interpolate the spectrum, in order to
        match the results of ``comodulogram``.

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    method : string in ('firstlast', 'minmax')
        Method for extracting a PAC metric from a DAR model

    fill : int in (0, 1, 2, 3, 4)
        Method to fill the spectral gap when removing the low frequencies

    ordar : int
        Order of the AR model used for whitening

    enf : float
        Electric network frequency, that will be removed

    random_noise : array or None, shape (n_points)
        Noise to be used to in the filling strategy

    normalize : boolean
        If True, the filtered signal is normalized

    whitening : boolean
        If True, the filtered signal is whitened

    progress_bar : boolean
        If True, a progress bar is shown in stdout

    n_surrogates : int
        Number of surrogates computed for the z-score

    minimum_shift : float
        Minimum time shift (in sec) for the surrogate analysis

    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the surrogate analysis

    Return
    ------
    comod : array, shape (len(low_fq_range), len(high_fq_range))
        Comodulogram for each couple of frequencies
    """
    sigdriv_imag = None
    low_sig = np.atleast_2d(low_sig)
    if high_sig is None:
        sigs = low_sig
    else:
        # hack to call only once extract
        high_sig = np.atleast_2d(high_sig)
        sigs = np.r_[low_sig, high_sig]
        n_epochs = low_sig.shape[0]

    sigs = np.atleast_2d(sigs)

    multiple_masks = (isinstance(mask, list) or
                      (isinstance(mask, np.ndarray) and mask.ndim == 3))
    if not multiple_masks:
        mask = [mask]

    extract_complex = model.ordriv_d > 0

    comod_list = None
    if progress_bar:
        bar = ProgressBar(
            max_value=len(low_fq_range) * len(mask),
            title='comodulogram: %s' % model.get_title(name=True))
    for j, filtered_signals in enumerate(extract(
            sigs=sigs, fs=fs, low_fq_range=low_fq_range,
            bandwidth=low_fq_width, fill=fill, ordar=ordar, enf=enf,
            random_noise=random_noise, normalize=normalize,
            whitening=whitening, draw='', extract_complex=extract_complex)):

        fc = low_fq_range[j]

        if extract_complex:
            filtered_low, filtered_high, filtered_low_imag = filtered_signals
        else:
            filtered_low, filtered_high = filtered_signals

        if high_sig is None:
            sigin = np.array(filtered_high)
            sigdriv = np.array(filtered_low)
            if extract_complex:
                sigdriv_imag = np.array(filtered_low_imag)
        else:
            sigin = np.array(filtered_high[n_epochs:])
            sigdriv = np.array(filtered_low[:n_epochs])
            if extract_complex:
                sigdriv_imag = np.array(filtered_low_imag[:n_epochs])

        sigin /= np.std(sigin)
        n_epochs, n_points = sigdriv.shape

        for i_mask, this_mask in enumerate(mask):
            def comod_function(shift):
                return _one_driven_modulation_index(fs, sigin, sigdriv,
                                                    sigdriv_imag, model,
                                                    this_mask, method,
                                                    high_fq_range, fc, shift)

            comod = _surrogate_analysis(comod_function, fs, n_points,
                                        minimum_shift, random_state,
                                        n_surrogates)

            # initialize the comodulogram arrays
            if comod_list is None:
                comod_list = []
                for _ in mask:
                    comod_list.append(np.zeros((low_fq_range.size,
                                                comod.size)))
            comod_list[i_mask][j, :] = comod

            if progress_bar:
                bar.update_with_increment_value(1)

    if not multiple_masks:
        return comod_list[0]
    else:
        return comod_list


def _one_driven_modulation_index(fs, sigin, sigdriv, sigdriv_imag, model, mask,
                                 method, high_fq_range, fc, shift):

    # shift for the surrogate analysis
    if shift != 0:
        sigdriv = np.roll(sigdriv, shift)

    # fit the model DAR on the data
    model.fit(fs=fs, sigin=sigin, sigdriv=sigdriv, sigdriv_imag=sigdriv_imag,
              mask=mask)

    # get PSD difference
    spec, _ = model.amplitude_frequency(fc=fc)
    if method == 'minmax':
        spec_diff = spec.max(axis=1) - spec.min(axis=1)
    elif method == 'firstlast':
        if model.ordriv_d_ == 0:
            # min and max of driver's values
            spec_diff = np.abs(spec[:, -1] - spec[:, 0])
        else:
            # largest PSD difference with a phase difference of np.pi
            n_freq, n_phases = spec.shape
            spec_diff = np.abs(spec - np.roll(spec, n_phases // 2, axis=1))
            _, j = argmax_2d(spec_diff)
            spec_diff = spec_diff[:, j]

    # crop the spectrum to high_fq_range
    frequencies = np.linspace(0, fs // 2, spec_diff.size)
    spec_diff = np.interp(high_fq_range, frequencies, spec_diff)

    return spec_diff


def _get_shifts(random_state, n_points, minimum_shift, fs, n_iterations):
    """ Compute the shifts for the surrogate analysis"""
    n_minimum_shift = max(1, int(fs * minimum_shift))
    # shift at least minimum_shift seconds, i.e. n_minimum_shift points
    if n_iterations > 1:
        if n_points - n_minimum_shift < n_minimum_shift:
            raise ValueError("The minimum shift is longer than the "
                             "visible data.")

        shifts = random_state.randint(
            n_minimum_shift, n_points - n_minimum_shift, size=n_iterations)
    else:
        shifts = np.array([0])

    # the first has no shift since this is for the initial computation
    shifts[0] = 0

    return shifts


def _surrogate_analysis(comod_function, fs, n_points, minimum_shift,
                        random_state, n_surrogates):
    """Call the comod function for several random time shifts,
    then compute the z-score of the result distribution."""
    # number of  surrogates MIs
    n_iterations = max(1, 1 + n_surrogates)

    # pre compute all the random time shifts
    shifts = _get_shifts(random_state, n_points, minimum_shift, fs,
                         n_iterations)

    comod_list = []
    for s, shift in enumerate(shifts):
        comod_list.append(comod_function(shift))
    comod_list = np.array(comod_list)

    # the first has no shift
    comod = comod_list[0, ...]

    # here we compute the z-score
    if n_iterations > 2:
        comod -= np.mean(comod_list[1:, ...], axis=0)
        comod /= np.std(comod_list[1:, ...], axis=0)

    return comod


def get_maximum_pac(comodulograms, low_fq_range, high_fq_range):
    """Get maximum PAC value in a comodulogram.
    'low_fq_range' and 'high_fq_range' must be the same than used in the
    modulation_index function that computed 'comodulogram'.

    Parameters
    ----------
    comodulograms : PAC values, shape (len(low_fq_range), len(high_fq_range))
                    If a list or a 3D array is given, it returns an array of
                    each value for each comodulogram.
    low_fq_range  : low frequency range (phase signal)
    high_fq_range : high frequency range (amplitude signal)

    Return
    ------
    low_fq    : low frequency of maximum PAC
    high_fq   : high frequency of maximum PAC
    pac_value : maximum PAC value
    """
    if isinstance(comodulograms, list):
        comodulograms = np.array(comodulograms)

    # only one comodulogram
    return_array = True
    if comodulograms.ndim == 2:
        comodulograms = comodulograms[None, :, :]
        return_array = False

    # check that the sizes match
    n_comod, n_low, n_high = comodulograms.shape
    n_low_2, n_high_2 = len(low_fq_range), len(high_fq_range)
    if n_low_2 != n_low or n_high_2 != n_high:
        raise ValueError("Array shapes do not match: (%d, %d) and (%d, %d)" %
                         (n_low, n_high, n_low_2, n_high_2))

    # compute the maximum of the comodulogram, and get the frequencies
    max_pac_value = np.zeros(n_comod)
    low_fq = np.zeros(n_comod)
    high_fq = np.zeros(n_comod)
    for k, comodulogram in enumerate(comodulograms):
        i, j = argmax_2d(comodulogram)
        max_pac_value[k] = comodulogram[i, j]

        low_fq[k] = low_fq_range[i]
        high_fq[k] = high_fq_range[j]

    # return arrays or floats
    if return_array:
        return low_fq, high_fq, max_pac_value
    else:
        return low_fq[0], high_fq[0], max_pac_value[0]
