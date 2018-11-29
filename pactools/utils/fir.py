import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from .spectrum import Spectrum


class FIR(object):
    """FIR filter

    Parameters
    ----------
    fir : array
        Finite impulse response (FIR) filter

    Examples
    --------
    >>> from pactools.utils.fir import FIR
    >>> f = FIR(fir=[0.2, 0.6, 0.2])
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fir=np.ones(1), fs=1.0):
        self.fir = fir
        self.fs = fs

    def transform(self, sigin, out=None, out_imag=None):
        """Apply this filter to a signal

        Parameters
        ----------
        sigin : array, shape (n_points, ) or (n_signals, n_points)
            Input signal

        Returns
        -------
        out : array, shape (n_points, ) or (n_signals, n_points)
            Filtered signal
        """
        sigin_ndim = sigin.ndim
        sigin = np.atleast_2d(sigin)

        if out is None:
            out = np.empty(sigin.shape)
        else:
            out = np.atleast_2d(out)
            assert out.dtype.kind in 'fc'

        for i, sig in enumerate(sigin):
            tmp = signal.fftconvolve(sig, self.fir, 'same')
            out[i] = tmp

        if sigin_ndim == 1:
            out = out[0]
        else:
            out = np.asarray(out)

        return out

    def plot(self, axs=None, fscale='log', colors=None):
        """
        Plots the impulse response and the transfer function of the filter.
        """
        # validate figure
        fig_passed = axs is not None
        if axs is None:
            fig, axs = plt.subplots(nrows=2)
        else:
            axs = np.atleast_1d(axs)
            if np.any([not isinstance(ax, plt.Axes) for ax in axs]):
                raise TypeError('axs must be a list of matplotlib Axes, got {}'
                                ' instead.'.format(type(axs)))
            # test if is figure and has 2 axes
            if len(axs) < 2:
                raise ValueError('Passed figure must have at least two axes'
                                 ', given figure has {}.'.format(len(axs)))
            fig = axs[0].figure

        # compute periodogram
        fft_length = max(int(2 ** np.ceil(np.log2(self.fir.shape[0]))), 2048)
        s = Spectrum(fft_length=fft_length, block_length=self.fir.size,
                     step=None, fs=self.fs, wfunc=np.ones, donorm=False)
        s.periodogram(self.fir)
        s.plot('Transfer function of FIR filter', fscale=fscale, axes=axs[0],
               colors=colors)

        # plots
        axs[1].plot(self.fir, color=colors[0])
        axs[1].set_title('Impulse response of FIR filter')
        axs[1].set_xlabel('Samples')
        axs[1].set_ylabel('Amplitude')
        if not fig_passed:
            fig.tight_layout()
        return fig


class BandPassFilter(FIR):
    """Band-pass FIR filter

    Designs a band-pass FIR filter centered on frequency fc.

    Parameters
    ----------
    fs : float
        Sampling frequency

    fc : float
        Center frequency of the bandpass filter

    n_cycles : float or None, (default 7.0)
        Number of oscillation in the wavelet. None if bandwidth is used.

    bandwidth : float or None, (default None)
        Bandwidth of the FIR wavelet filter. None if n_cycles is used.

    zero_mean : boolean, (default True)
        If True, the mean of the FIR is subtracted, i.e. fir.sum() = 0.

    extract_complex : boolean, (default False)
        If True, the wavelet filter is complex and ``transform`` returns two
        signals, filtered with the real and the imaginary part of the filter.

    Examples
    --------
    >>> from pactools.utils import BandPassFilter
    >>> f = BandPassFilter(fs=100., fc=5., bandwidth=1., n_cycles=None)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fs, fc, n_cycles=7.0, bandwidth=None, zero_mean=True,
                 extract_complex=False):
        self.fc = fc
        self.fs = fs
        self.n_cycles = n_cycles
        self.bandwidth = bandwidth
        self.zero_mean = zero_mean
        self.extract_complex = extract_complex

        self._design()

    def _design(self):
        """Designs the FIR filter"""
        # the length of the filter
        order = self._get_order()
        half_order = (order - 1) // 2

        w = np.blackman(order)
        t = np.linspace(-half_order, half_order, order)
        phase = (2.0 * np.pi * self.fc / self.fs) * t

        car = np.cos(phase)
        fir = w * car

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        # remove the constant component by forcing fir.sum() = 0
        if self.zero_mean:
            fir -= fir.sum() / order

        gain = np.sum(fir * car)
        self.fir = fir * (1.0 / gain)

        # add the imaginary part to have a complex wavelet
        if self.extract_complex:
            car_imag = np.sin(phase)
            fir_imag = w * car_imag
            self.fir_imag = fir_imag * (1.0 / gain)
        return self

    def _get_order(self):
        if self.bandwidth is None and self.n_cycles is not None:
            half_order = int(float(self.n_cycles) / self.fc * self.fs / 2)
        elif self.bandwidth is not None and self.n_cycles is None:
            half_order = int(1.65 * self.fs / self.bandwidth) // 2
        else:
            raise ValueError('fir.BandPassFilter: n_cycles and bandwidth '
                             'cannot be both None, or both not None. Got '
                             '%s and %s' % (self.n_cycles, self.bandwidth, ))

        order = half_order * 2 + 1
        return order

    def transform(self, sigin, out=None, out_imag=None):
        """Apply this filter to a signal

        Parameters
        ----------
        sigin : array, shape (n_points, ) or (n_signals, n_points)
            Input signal

        Returns
        -------
        filtered : array, shape (n_points, ) or (n_signals, n_points)
            Filtered signal

        (filtered_imag) : array, shape (n_points, ) or (n_signals, n_points)
            Only when extract_complex is true.
            Filtered signal with the imaginary part of the filter
        """
        filtered = super(BandPassFilter, self).transform(sigin, out=out)

        if self.extract_complex:
            fir = FIR(fir=self.fir_imag, fs=self.fs)
            filtered_imag = fir.transform(sigin, out=out_imag)
            return filtered, filtered_imag
        else:
            return filtered

    def plot(self, axs=None, fscale='log', colors=None):
        """
        Plots the impulse response and the transfer function of the filter.
        """
        fig = super(BandPassFilter, self).plot(axs=axs, fscale=fscale,
                                               colors=colors)

        if self.extract_complex:
            if axs is None:
                axs = fig.axes

            fir = FIR(fir=self.fir_imag, fs=self.fs)
            fir.plot(axs=axs, fscale=fscale, colors=colors)
        return fig


class LowPassFilter(FIR):
    """Low-pass FIR filter

    Designs a FIR filter that is a low-pass filter.

    Parameters
    ----------
    fs : float
        Sampling frequency

    fc : float
        Cut-off frequency of the low-pass filter

    bandwidth : float
        Bandwidth of the FIR wavelet filter

    ripple_db : float (default 60.0)
        Positive number specifying maximum ripple in passband (dB) and minimum
        ripple in stopband, in Kaiser-window low-pass FIR filter.

    Examples
    --------
    >>> from pactools.utils import LowPassFilter
    >>> f = LowPassFilter(fs=100., fc=5., bandwidth=1.)
    >>> f.plot()
    >>> signal_out = f.transform(signal_in)
    """

    def __init__(self, fs, fc, bandwidth, ripple_db=60.0):
        self.fs = fs
        self.fc = fc
        self.bandwidth = bandwidth
        self.ripple_db = ripple_db

        self._design()

    def _design(self):
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(self.ripple_db,
                                   self.bandwidth / self.fs * 2)

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        fir = signal.firwin(N, self.fc / self.fs * 2, window=('kaiser', beta))

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        self.fir = fir / np.sum(fir)
        return self
