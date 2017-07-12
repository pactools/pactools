import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from .spectrum import Spectrum
from .deprecation import deprecated


@deprecated("Please use pactools.utils.fir.BandPassFilter instead")
class Carrier(object):
    def __init__(self, fir=np.ones(1), fs=1., extract_complex=False):
        self.fir = fir
        self.fs = fs
        self.extract_complex = extract_complex

    def design(self, fs, fc, n_cycles=7.0, bandwidth=None, zero_mean=True):
        """
        Designs a FIR filter that is a band-pass filter centered
        on frequency fc.
        fs         : sampling frequency (Hz)
        fc         : frequency of the carrier (Hz)
        n_cycles   : number of oscillation in the wavelet
        bandwidth  : bandwidth of the FIR wavelet filter
                     (used when: bandwidth is not None and n_cycles is None)
        zero_mean  : if True, we make sure fir.sum() = 0
        """
        # the length of the filter
        if bandwidth is None and n_cycles is not None:
            half_order = int(float(n_cycles) / fc * fs / 2)
            order = 2 * half_order + 1
        elif bandwidth is not None and n_cycles is None:
            half_order = int(1.65 * fs / bandwidth) // 2
            order = half_order * 2 + 1
        else:
            raise ValueError('Carrier.design: n_cycles and order cannot be '
                             'both None or both not None.')

        w = np.blackman(order)
        t = np.linspace(-half_order, half_order, order)
        phase = (2.0 * np.pi * fc / fs) * t

        car = np.cos(phase)
        fir = w * car

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        # remove the constant component by forcing fir.sum() = 0
        if zero_mean:
            fir -= fir.sum() / order

        gain = np.sum(fir * car)
        self.fir = fir * (1.0 / gain)
        self.fs = fs

        # add the imaginary part to have a complex wavelet
        if self.extract_complex:
            car_imag = np.sin(phase)
            fir_imag = w * car_imag
            self.fir_imag = fir_imag * (1.0 / gain)

        return self

    def plot(self, fig=None, fscale='log', print_width=False):
        """
        Plots the impulse response and the transfer function of the filter.
        """
        # validate figure
        if fig is None:
            fig_passed = False
            fig, axes = plt.subplots(nrows=2)
        else:
            fig_passed = True
            if not isinstance(fig, plt.Figure):
                raise TypeError('fig must be matplotlib Figure, got {}'
                                ' instead.'.format(type(fig)))
            # test if is figure and has 2 axes
            n_axes = len(fig.axes)
            if n_axes < 2:
                raise ValueError('Passed figure must have at least two axes'
                                 ', given figure has {}.'.format(n_axes))
            axes = fig.axes

        # compute periodogram
        fft_length = max(int(2 ** np.ceil(np.log2(self.fir.shape[0]))), 1024)
        s = Spectrum(fft_length=fft_length, block_length=self.fir.size,
                     step=None, fs=self.fs, wfunc=np.ones, donorm=False)
        s.periodogram(self.fir)
        s.plot('Transfer function of FIR filter "Carrier"', fscale=fscale,
               axes=axes[0])

        # print the frequency and the bandwidth
        if print_width:
            freq = np.linspace(0, s.fs / 2, s.fft_length // 2 + 1)

            psd = s.psd[0][0, :]
            psd = 10.0 * np.log10(np.maximum(psd, 1.0e-12))
            i0 = np.argmax(psd)
            over_3db = np.nonzero(psd > psd[i0] - 3)[0]

            f_low = freq[over_3db[0]]
            f_high = freq[over_3db[-1]]
            df = f_high - f_low
            f0 = freq[i0]
            print('f0 = %.3f Hz, df_3db = %.3f Hz [%.3f, %.3f], df/f = %.3f' %
                  (f0, df, f_low, f_high, df / f0))

        # plots
        axes[1].plot(self.fir)
        if self.extract_complex:
            axes[1].plot(self.fir_imag)
        axes[1].set_title('Impulse response of FIR filter "Carrier"')
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Amplitude')
        if not fig_passed:
            fig.tight_layout()
        return fig

    def direct(self, sigin):
        """
        apply this filter to a signal
        sigin : input signal (ndarray)
        returns the filtered signal (ndarray)
        """
        fftconvolve = signal.fftconvolve

        filtered = fftconvolve(sigin.ravel(), self.fir, 'same')
        if self.extract_complex:
            filtered_imag = fftconvolve(sigin.ravel(), self.fir_imag, 'same')

        if sigin.ndim == 2:
            filtered = filtered[None, :]
            if self.extract_complex:
                filtered_imag = filtered_imag[None, :]

        if self.extract_complex:
            return filtered, filtered_imag
        else:
            return filtered


@deprecated("Please use pactools.utils.fir.LowPassFilter instead")
class LowPass(Carrier):
    def __init__(self, fs=1.):
        self.fir = np.ones(1)
        self.fs = fs
        self.extract_complex = False

    def design(self, fs, fc, bandwidth, ripple_db=60.0):
        """
        Designs a FIR filter that is a low-pass filter.
        fs : sampling frequency (Hz)
        fc : cut-off frequency (Hz)
        bandwidth : transition bandwidth (Hz)s
        """
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = signal.kaiserord(ripple_db, bandwidth / fs * 2)

        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        fir = signal.firwin(N, fc / fs * 2, window=('kaiser', beta))

        # the filter must be symmetric, in order to be zero-phase
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        self.fir = fir / np.sum(fir)
        self.fs = fs
        return self
