#!/usr/bin/python
"""
filter that extracts a low frequency carrier
"""
from __future__ import print_function

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from .spectrum import Spectrum


class Carrier:
    def __init__(self, fir=np.ones(1), fs=1.):
        """
        initializes a filter with coefficient in a (ndarray)
        """
        self.fir = fir
        self.fs = fs

    def design(self, fs, fc, n_cycles=7.0, bandwidth=None, zero_mean=True):
        """
        designs a FIR filter that is a band-pass filter centered
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
        car = np.cos((2.0 * np.pi * fc / fs) * t)
        fir = w * car

        # the filter must be odd and symmetric, in order to be zero-phase
        assert fir.size % 2 == 1
        assert np.all(np.abs(fir - fir[::-1]) < 1e-15)

        # remove the constant component by forcing fir.sum() = 0
        if zero_mean:
            fir -= fir.sum() / order

        gain = np.sum(fir * car)
        self.fir = fir * (1.0 / gain)
        self.fs = fs
        return self

    def plot(self, fig=None, fscale='log', print_width=False):
        """
        plots the impulse response and the transfer function of the filter
        """
        # compute periodogram
        nfft = max(int(2 ** np.ceil(np.log2(self.fir.shape[0]))), 1024)
        s = Spectrum(fftlen=nfft,
                     blklen=self.fir.size,
                     step=np.nan,
                     fs=self.fs,
                     wfunc=np.ones,
                     donorm=False)
        s.periodogram(self.fir)
        s.plot('Transfer function of FIR filter "Carrier"', fscale=fscale)

        # print the frequency and the bandwidth
        if print_width:
            freq = np.linspace(0, s.fs / 2, s.fftlen // 2 + 1)

            psd = s.psd[0][0, 0:s.fftlen // 2 + 1]
            psd = 10.0 * np.log10(np.maximum(psd, 1.0e-12))
            i0 = np.argmax(psd)
            over_3db = np.nonzero(psd > psd[i0] - 3)[0]

            f_low = freq[over_3db[0]]
            f_high = freq[over_3db[-1]]
            df = f_high - f_low
            f0 = freq[i0]
            print('f0 = %.3f Hz, df_3db = %.3f Hz, df/f = %.3f'
                  % (f0, df, df / f0))

        # plots
        if fig is None:
            fig = plt.figure('Impulse response of FIR filter "Carrier"')
        ax = fig.gca()
        ax.plot(self.fir)
        ax.set_title('Impulse response of FIR filter "Carrier"')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        return fig

    def direct(self, sigin):
        """
        apply this filter to a signal
        sigin : input signal (ndarray)
        returns the filtered signal (ndarray)
        """
        if sigin.ndim == 2:
            return signal.fftconvolve(sigin[0], self.fir, 'same')[None, :]
        else:
            return signal.fftconvolve(sigin, self.fir, 'same')


def test1():
    print('testing carrier with constant n_cycles')
    fs = 33.83

    fig = None
    carrier = Carrier()
    for f in np.linspace(1.0, 15.0, 10):
        carrier.design(fs, f, n_cycles=7.0)
        fig = carrier.plot(fig=fig, fscale='lin', print_width=True)

    plt.show()

    print('testing carrier with constant bandwidth')
    fig = None
    carrier = Carrier()
    for f in np.linspace(1.0, 15.0, 10):
        carrier.design(fs, f, n_cycles=None, bandwidth=1.1785, zero_mean=True)
        fig = carrier.plot(fig=fig, fscale='lin', print_width=True)

    plt.show()
