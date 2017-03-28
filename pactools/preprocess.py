import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from .utils.spectrum import Spectrum
from .utils.carrier import Carrier, LowPass
from .utils.arma import Arma
from .utils.validation import check_random_state


def _decimate(x, q):
    """
    Downsample the signal after low-pass filtering to avoid aliasing.
    An order 16 Chebyshev type I filter is used.

    Parameters
    ----------
    x : ndarray
        The signal to be downsampled, as an N-dimensional array.
    q : int
        The downsampling factor.

    Returns
    -------
    y : ndarray
        The down-sampled signal.
    """
    if not isinstance(q, int):
        raise (TypeError, "q must be an integer")

    b, a = signal.filter_design.cheby1(16, 0.025, 0.98 / q)

    y = signal.filtfilt(b, a, x, axis=-1)

    sl = [slice(None)] * y.ndim
    sl[-1] = slice(None, None, q)
    return y[sl]


def decimate(sig, fs, decimation_factor):
    """Decimates the signal:
    Downsampling after low-pass filtering to avoid aliasing

    Parameters
    ----------
    sig : array
        Raw input signal
    fs : float
        Sampling frequency of the input
    decimation_factor : int > 0
        Ratio of sampling frequencies (old/new)

    Returns
    -------
    sig : array
        Decimated signal
    fs : float
        Sampling frequency of the output

    """
    # -------- center the signal
    sig = sig - np.mean(sig)

    # -------- resample
    # decimation could be performed in two steps for better performance
    # 0 in the following array means no decimation
    dec_1st = [
        0, 0, 2, 3, 4, 5, 6, 7, 2, 3, 2, 0, 3, 0, 2, 3, 4, 0, 3, 0, 4, 3, 0, 0,
        4, 5, 0, 3, 4, 0, 5
    ]
    dec_2nd = [
        0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 5, 0, 4, 0, 7, 5, 4, 0, 6, 0, 5, 7, 0, 0,
        6, 5, 0, 9, 7, 0, 6
    ]

    d1 = dec_1st[decimation_factor]
    if d1 == 0:
        raise (ValueError, 'cannot decimate by %d' % decimation_factor)

    sig = _decimate(sig, d1)
    sig = sig.astype(np.float32)
    d2 = dec_2nd[decimation_factor]
    if d2 > 0:
        sig = _decimate(sig, d2)
        sig = sig.astype(np.float32)

    fs = fs / decimation_factor

    return sig, fs


def extract_and_fill(sig, fs, fc, n_cycles=None, bandwidth=1.0, fill=0,
                     draw='', ordar=8, enf=50.0, whiten_fill4=True,
                     random_noise=None, extract_complex=True, low_pass=False,
                     random_state=None):
    """Creates a FIR bandpass filter, applies this filter to a signal to obtain
    the filtered signal low_sig and its complement high_sig.
    Also fills the frequency gap in high_sig.

    sig : array
        Input signal
    fs : float
        Sampling frequency
    fc : float
        Center frequency of the bandpass filter
    n_cycles : float
        Number of cycles in the bandpass filter
        Should be None if bandwidth is not None
    bandwidth : float
        Bandwidth of the bandpass filter
        Should be None if n_cycles is not None
    fill : int in {0, 1, 2, 3, 4, 5}
        Filling strategy for in high_sig
        0 : keep the signal unchanged: high_sig = sig
        1 : remove (the bandpass filtered signal): high_sig = sig - low_sig
        2 : remove and replace by bandpass filtered Gaussian white noise
        3 : remove and replace by low_sig[::-1]
        4 : removing a wide-band around carrier and then use fill=2
        5 : remove and replace by low_sig[::-1]
    draw : string
        List of plots to draw
    ordar : int > 0
        AR order for the whitening (only used when fill=4)
    extract_complex : boolean
        Use a complex wavelet
    low_pass : boolean
        Use a lowpass filter at fc instead of a bandpass filter centered at fc
    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the surrogate analysis

    Returns
    -------
    low_sig : array
        Bandpass filtered signal
    high_sig : array
        Processed fullband signal
    low_sig_imag : array (returned only if extract_complex is True)
        Imaginary part of the bandpass filtered signal

    """
    rng = check_random_state(random_state)
    if random_noise is None:
        random_noise = rng.randn(len(sig))

    if low_pass:
        filt = LowPass().design(fs=fs, fc=fc, bandwidth=bandwidth)
        if extract_complex:
            raise NotImplementedError('extract_complex incompatible with '
                                      'low_pass filter.')
    else:
        filt = Carrier(extract_complex=extract_complex)
        filt.design(fs, fc, n_cycles, bandwidth, zero_mean=True)
        if 'c' in draw or 'z' in draw:
            filt.plot(fscale='lin', print_width=True)

    if extract_complex:
        low_sig, low_sig_imag = filt.direct(sig)
    else:
        low_sig = filt.direct(sig)

    if fill == 0:
        # keeping driver in high_sig
        high_sig = sig

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum([sig, low_sig, high_sig], labels=None,
                                    fs=fs, colors='bggck')
            plt.legend(['signal', 'driver', 'high_frequencies'], loc=0)

    elif fill == 1:
        # subtracting driver
        high_sig = sig - low_sig

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum([sig, low_sig, sig - low_sig, high_sig],
                                    labels=None, fs=fs, colors='bggck')
            plt.legend(
                ['signal', 'driver', 'signal-driver',
                 'high_frequencies'], loc=0)

    elif fill == 2:
        # replacing driver by a white noise
        if extract_complex:
            fill_sig, _ = filt.direct(random_noise)
        else:
            fill_sig = filt.direct(random_noise)
        fill_sig.shape = sig.shape
        fill_sig *= np.std(low_sig) / np.std(fill_sig)

        high_sig = sig - low_sig + fill_sig

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum(
                [sig, low_sig, sig - low_sig, fill_sig, high_sig], labels=None,
                fs=fs, colors='bggck')
            plt.legend([
                'signal', 'driver', 'signal-driver', 'noise',
                'high_frequencies'
            ], loc=0)

    elif fill == 3:
        # 'replacing with driver[::-1]
        fill_sig = low_sig.ravel()[::-1]
        fill_sig.shape = sig.shape
        high_sig = sig - low_sig + fill_sig

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum(
                [sig, low_sig, sig - low_sig, fill_sig, high_sig], labels=None,
                fs=fs, colors='bggck')
            plt.legend([
                'signal', 'driver', 'signal-driver', 'driver[::-1]',
                'high_frequencies'
            ], loc=0)

    elif fill == 4:
        # replacing driver by a wide-band white noise

        factor = 8.0
        if n_cycles is not None:
            n_cycles /= factor
        else:
            bandwidth *= factor

        # apply whitening here to have a better filling
        if whiten_fill4:
            white_sig = whiten(sig, fs, ordar, draw=draw, enf=enf)
        else:
            white_sig = sig

        wide_low_sig, high_sig = extract_and_fill(
            sig=white_sig, fs=fs, fc=fc, n_cycles=n_cycles,
            bandwidth=bandwidth, fill=2, draw=draw, enf=enf,
            random_noise=random_noise, extract_complex=False,
            low_pass=low_pass, random_state=rng)

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum([
                sig, low_sig, sig - low_sig, white_sig, wide_low_sig,
                white_sig - wide_low_sig, high_sig
            ], labels=None, fs=fs, colors='bggcrrk')
            plt.legend([
                'signal', 'driver', 'signal-driver', 'white_signal',
                'wide_driver', 'white_signal-wide_driver', 'high_frequencies'
            ], loc=0)
    elif fill == 5:
        high_sig = sig - low_sig

        if extract_complex:
            fill_sig, _ = filt.direct(random_noise)
        else:
            fill_sig = filt.direct(random_noise)
        fill_sig.shape = sig.shape

        high_sig = fill_gap(high_sig, fs, fa=fc, dfa=bandwidth, draw=draw,
                            fill_sig=fill_sig)

        if 'z' in draw or 'c' in draw:
            _plot_multiple_spectrum([sig, low_sig, sig - low_sig, high_sig],
                                    labels=None, fs=fs, colors='bgcr')
            plt.legend(
                ['signal', 'driver', 'signal-driver',
                 'high_frequencies'], loc=0)
    else:
        raise (ValueError, 'Invalid fill parameter: %s' % str(fill))

    if extract_complex:
        return low_sig, high_sig, low_sig_imag
    else:
        return low_sig, high_sig


def low_pass_and_fill(sig, fs, fc=1.0, draw='', bandwidth=1.,
                      random_state=None):
    low_sig, high_sig = extract_and_fill(sig, fs, fc, fill=1, low_pass=True,
                                         bandwidth=bandwidth,
                                         random_state=random_state)

    rng = check_random_state(random_state)
    random_noise = rng.randn(*sig.shape)

    filt = LowPass().design(fs=fs, fc=fc)
    fill_sig = filt.direct(random_noise)
    filled_sig = fill_gap(sig=high_sig, fs=fs, fa=fc / 2., dfa=fc / 2.,
                          draw=draw, fill_sig=fill_sig)
    return filled_sig


def _plot_multiple_spectrum(signals, fs, labels, colors):
    """
    plot the signals spectrum
    """
    s = Spectrum(block_length=min(2048, signals[0].size), fs=fs,
                 wfunc=np.blackman)
    for sig in signals:
        s.periodogram(sig, hold=True)
    s.plot(labels=labels, colors=colors, fscale='lin')


def whiten(sig, fs, ordar=8, draw='', enf=50.0, d_enf=1.0, zero_phase=True,
           **kwargs):
    """Use an AR model to whiten a signal
    The whitening filter is not estimated around multiples of
    the electric network frequency (up to d_enf Hz)

    sig   : input signal
    fs    : sampling frequency of input signal
    ordar : order of AR whitening filter
    draw  : list of plots
    enf   : electric network frequency
    denf  : tolerance on electric network frequency
    zero_phase : if True, apply half the whitening for sig(t) and sig(-t)

    returns the whitened signal

    """
    # -------- create the AR model and its spectrum
    ar = Arma(ordar=ordar, ordma=0, fs=fs, block_length=min(1024, sig.size))
    ar.periodogram(sig)
    # duplicate to see the removal of the electric network frequency
    ar.periodogram(sig, hold=True)
    fft_length, _ = ar.check_params()

    # -------- remove the influence of the electric network frequency
    k = 1
    # while the harmonic k is included in the spectrum
    while k * enf - d_enf < fs / 2.0:
        fmin = k * enf - d_enf
        fmax = k * enf + d_enf
        kmin = max((0, int(fft_length * fmin / fs)))
        kmax = min(fft_length // 2, int(fft_length * fmax / fs) + 1)
        Amin = ar.psd[-1][0, kmin]
        Amax = ar.psd[-1][0, kmax]
        # linear interpolation between (kmin, Amin) and (kmax, Amax)
        interpol = np.linspace(Amin, Amax, kmax - kmin, endpoint=False)

        # remove positive frequencies
        ar.psd[-1][0, kmin:kmax] = interpol

        k += 1

    # -------- change psd for zero phase filtering
    if zero_phase:
        ar.psd[-1] = np.sqrt(ar.psd[-1])

    # -------- estimate the model and apply it
    ar.estimate()

    # apply the whitening twice (forward and backward) for zero-phase filtering
    if zero_phase:
        sigout = ar.inverse(sig)
        sigout = sigout[::-1]
        sigout = ar.inverse(sigout)
        sigout = sigout[::-1]
    else:
        sigout = ar.inverse(sig)

    gain = np.std(sig) / np.std(sigout)
    sigout *= gain

    if 'w' in draw or 'z' in draw:
        ar.arma2psd(hold=True)
        ar.periodogram(sigout, hold=True)
        ar.plot('periodogram before/after whitening', labels=[
            'with electric network', 'without electric network', 'model AR',
            'whitened'
        ], fscale='lin')
        plt.legend(loc='lower left')

    return sigout


def fill_gap(sig, fs, fa=50.0, dfa=25.0, draw='', fill_sig=None,
             random_state=None):
    """Fill a gap with white noise.
    """
    rng = check_random_state(random_state)

    # -------- get the amplitude of the gap
    sp = Spectrum(block_length=min(2048, sig.size), fs=fs, wfunc=np.blackman)
    fft_length, _ = sp.check_params()
    sp.periodogram(sig)
    fmin = fa - dfa
    fmax = fa + dfa
    kmin = max((0, int(fft_length * fmin / fs)))
    kmax = min(fft_length // 2, int(fft_length * fmax / fs) + 1)
    Amin = sp.psd[-1][0, kmin]
    Amax = sp.psd[-1][0, kmax]

    if kmin == 0 and kmax == (fft_length // 2):
        # we can't fill the entire spectrum
        return sig
    if kmin == 0:
        # if the gap reach zero, we only consider the right bound
        Amin = Amax
    if kmax == (fft_length // 2):
        # if the gap reach fft_length / 2, we only consider the left bound
        Amax = Amin

    A_fa = (Amin + Amax) * 0.5

    # -------- bandpass filtering of white noise
    if fill_sig is None:
        n_cycles = 1.65 * fa / dfa
        fir = Carrier()
        fir.design(fs, fa, n_cycles, None, zero_mean=False)
        fill_sig = fir.direct(rng.randn(*sig.shape))

    # -------- compute the scale parameter
    sp.periodogram(fill_sig, hold=True)
    kfa = int(fft_length * fa / fs)
    scale = np.sqrt(A_fa / sp.psd[-1][0, kfa])
    fill_sig *= scale

    sig += fill_sig
    if 'g' in draw or 'z' in draw:
        labels = ['signal', 'fill signal', 'gap filled']
        sp.periodogram(sig, hold=True)
        sp.plot(labels=labels, fscale='lin', title='fill')

    return sig


def _show_plot(draw):
    if draw:
        plt.show()


def extract_driver(low_fq, *args, **kwargs):
    """helper"""
    for sigs in multiple_extract(low_fq_range=[low_fq], *args, **kwargs):
        return sigs


def multiple_extract(sigs, fs, low_fq_range, n_cycles=None, bandwidth=1.0,
                     fill=5, draw='', ordar=8, enf=50.0, random_noise=None,
                     normalize=False, whitening='after', extract_complex=True,
                     random_state=None):
    """
    Do fast preprocessing for several values of fc (the driver frequency).

    Example
    -------
    for (low_sig, high_sig) in extract_multi(sigs, fs, low_fq_range):
        pass
    """
    low_fq_range = np.atleast_1d(low_fq_range)
    sigs = np.atleast_2d(sigs)
    rng = check_random_state(random_state)

    if whitening == 'before':
        sigs = [whiten(sig, fs=fs, ordar=ordar, draw=draw) for sig in sigs]
        _show_plot(draw)

    if fill in [2, 4, 5]:
        random_noise = rng.randn(len(sigs[0]))
    else:
        random_noise = None

    # extract the high frequencies independently of the driver
    whiten_fill4 = whitening == 'after'
    fc_low_pass = low_fq_range[-1] + low_fq_range[0] + bandwidth  # hack
    low_pass_width = bandwidth
    low_and_high = [
        extract_and_fill(sig, fs=fs, fc=fc_low_pass, bandwidth=low_pass_width,
                         fill=fill, ordar=ordar, enf=enf,
                         whiten_fill4=whiten_fill4, random_noise=random_noise,
                         draw=draw, extract_complex=False, low_pass=True,
                         random_state=random_state) for sig in sigs
    ]
    high_sigs = [both[1] for both in low_and_high]

    if whitening == 'after' and fill != 4:
        high_sigs = [
            whiten(high_sig, fs=fs, ordar=ordar, draw=draw)
            for high_sig in high_sigs
        ]

    if normalize:
        scales = [1.0 / np.std(high_sig) for high_sig in high_sigs]
        high_sigs = [high * s for (high, s) in zip(high_sigs, scales)]

    # as high_sigs is now fixed, we don't need the following
    fill = 0
    random_noise = None
    whiten_fill4 = False

    # extract_and_fill the driver
    for fc in low_fq_range:
        low_and_high = [
            extract_and_fill(
                sig, fs=fs, fc=fc, n_cycles=n_cycles, bandwidth=bandwidth,
                fill=fill, ordar=ordar, enf=enf, whiten_fill4=whiten_fill4,
                random_noise=random_noise, draw=draw,
                extract_complex=extract_complex, random_state=random_state)
            for sig in sigs
        ]
        low_sigs = [both[0] for both in low_and_high]
        if extract_complex:
            low_sigs_imag = [both[2] for both in low_and_high]

        # normalize variances
        if normalize:
            low_sigs = [low * s for (low, s) in zip(low_sigs, scales)]
            if extract_complex:
                low_sigs_imag = [
                    low * s for (low, s) in zip(low_sigs_imag, scales)
                ]

        _show_plot(draw)

        if extract_complex:
            yield low_sigs, high_sigs, low_sigs_imag
        else:
            yield low_sigs, high_sigs
