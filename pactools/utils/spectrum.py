"""
Estimator of power spectral densities
"""
from __future__ import print_function
from itertools import chain

import numpy as np
import scipy as sp
from scipy.linalg import hankel
from scipy.signal import hilbert
import matplotlib.pyplot as plt


class Spectrum(object):
    def __init__(self, blklen=1024, fftlen=None, step=None, wfunc=np.hamming,
                 fs=1, donorm=True):
        """
        initalizes an estimator

        blklen : length of each signal block
        fftlen : length of FFT (it is expected that fftlen >= blklen)
                 set to blklen if None
        step   : step between successive blocks
                 set to blklen // 2 if None
        wfunc  : function used to compute weitghting function
        fs     : sampling frequency
        donorm : normalize amplitude if True

        """
        self.blklen = blklen
        self.fftlen = fftlen
        self.step = step
        self.wfunc = wfunc
        self.fs = fs
        self.donorm = donorm
        self.psd = []

    class Blklen(object):
        def __get__(self, instance, owner):
            return self.blklen

        def __set__(self, instance, value):
            if value < 0:
                raise ValueError(
                    'spectrum: block length is negative')
            self.blklen = value
    blklen = Blklen()

    class Fftlen(object):
        def __get__(self, instance, owner):
            if self.fftlen is None:
                return instance.blklen
            else:
                return self.fftlen

        def __set__(self, instance, value):
            if (value not in {2 ** x for x in range(4, 20)} and
               value is not None):
                raise ValueError(
                    'spectrum: FFT length should be in: 2**4, 2**5...2**20')
            if value is not None and value < instance.blklen:
                raise ValueError(
                    'spectrum: block length is greater than FFT length')
            self.fftlen = value

    fftlen = Fftlen()

    class Step(object):
        def __get__(self, instance, owner):
            if self.step is None:
                return instance.blklen // 2
            else:
                return self.step

        def __set__(self, instance, value):
            if value is not None and (value <= 0 or value > instance.blklen):
                raise ValueError(
                    'spectrum: invalid step between blocks')
            self.step = value
    step = Step()

    def periodogram(self, signals, hold=False):
        """
        computes the estimation (in dB) for one signal

        signals : signals from which one computes the power spectrum
                  (if self.donorm = True) or impulse response used
                  to compute a transfer function (set donorm to False)
        hold    : if True, the estimation is appended to the list of
                  previous estimations,
                  if False, the list is emptied and only the current
                  estimation is stored
        """
        if len(signals.shape) < 2:
            signals = signals[None, :]

        w = self.wfunc(self.blklen)
        n_signals, tmax = signals.shape
        psd = np.zeros((n_signals, self.fftlen))

        for i, sig in enumerate(signals):
            block = np.arange(self.blklen)

            # iterate on blocks
            count = 0
            while block[-1] < sig.size:
                psd[i] += np.abs(sp.fft(w * sig[block], self.fftlen, 0)) ** 2
                count = count + 1
                block = block + self.step
            if count == 0:
                raise IndexError(
                    'spectrum: first block has %d samples but sig has %d '
                    'samples' % (block[-1] + 1, sig.size))

            # normalize
            if self.donorm:
                scale = 1.0 / (count * (np.sum(w) ** 2))
            else:
                scale = 1.0 / count
            psd[i] *= scale

        if not hold:
            self.psd = []
        self.psd.append(psd)
        return psd

    def plot(self, title='', fscale='lin', labels=None, fig=None,
             replicate=None, colors=None):
        """
        plots the power spectral density
        warning: the plot will only appear after plt.show()

        returns the figure instance

        title  : title of the plot (and the window)
        fscale : kind of frequency scale ('lin' or 'log')
        labels : list of label of the plots
        replicate: number of replication of the spectrum across frequencies
        """
        if labels is None:
            plot_legend = False
            labels = [''] * len(self.psd)
        else:
            plot_legend = True
        if isinstance(labels, str):
            labels = [labels]

        if replicate is None:
            replicate = 0

        if colors is None:
            colors = ('bgrcmyk' * 100)[:len(self.psd)]

        if fig is None:
            fig = plt.figure(title).gca()
        try:
            fig = fig.gca()
        except:
            pass

        self.fscale = fscale
        if self.fscale == 'log':
            fig.set_xscale('log')
        else:
            fig.set_xscale('linear')

        fmax = self.fs / 2
        freq = np.linspace(0, fmax, self.fftlen // 2 + 1)

        if len(self.psd) > 1:
            plt.hold(True)
        for label_, color, psd in zip(labels, colors, self.psd):
            psd = 10.0 * np.log10(np.maximum(psd, 1.0e-16))
            for i in range(replicate + 1):
                label = label_ if i == 0 else ''
                fig.plot(freq + i * fmax,
                         psd[:, 0:self.fftlen // 2 + 1].T[::(-1) ** i],
                         label=label, color=color)

        fig.grid(True)
        fig.set_title(title)
        fig.set_xlabel('Frequency (Hz)')
        fig.set_ylabel('Amplitude (dB)')
        if plot_legend:
            fig.legend(loc=0)
        return fig

    def main_frequency(self):
        freq = np.linspace(0, self.fs / 2, self.fftlen // 2 + 1)
        psd = self.psd[-1][0, 0:self.fftlen // 2 + 1]
        return freq[np.argmax(psd)]


class Bicoherence(Spectrum):
    def __init__(self, blklen=1024, fftlen=None, step=None, wfunc=np.hamming,
                 fs=1):
        """
        initalizes an estimator

        blklen : length of each signal block
        fftlen : length of FFT (it is expected that fftlen >= blklen)
                 set to blklen if None
        step   : step between successive blocks
                 set to blklen // 2 if None
        wfunc  : function used to compute weitghting function
        fs     : sampling frequency

        """
        super(Bicoherence, self).__init__(
            blklen=blklen, fftlen=fftlen, step=step,
            wfunc=wfunc, fs=fs)

    def fit(self, sigs, method='hagihira'):
        """
        computes the estimation (in dB) for one signal

        sigs     : signal from which one computes the bicoherence
        """
        self.method = method

        sigs = np.atleast_2d(sigs)
        n_epochs, n_points = sigs.shape

        w = self.wfunc(self.blklen)
        n_freq = self.fftlen // 2 + 1
        bicoherence = np.zeros((n_freq, n_freq), dtype=np.complex128)
        normalization = np.zeros((n_freq, n_freq), dtype=np.float64)

        # iterate on blocks
        count = 0
        for i_epoch in range(n_epochs):
            block = np.arange(self.blklen)
            while block[-1] < n_points:
                F = sp.fft(w * sigs[i_epoch, block], self.fftlen, 0)[:n_freq]
                F1 = F[None, :]
                F2 = F1.T
                mask = hankel(np.arange(n_freq))
                F12 = np.conjugate(F)[mask]

                product = F1 * F2 * F12
                bicoherence += product
                if method == 'sigl':
                    normalization += square(F1) * square(F2) * square(F12)
                elif method == 'nagashima':
                    normalization += square(F1 * F2) * square(F12)
                elif method == 'hagihira':
                    normalization += np.abs(product)
                elif method == 'bispectrum':
                    pass
                else:
                    raise(ValueError("Method '%s' unkown." % method))

                count = count + 1
                block = block + self.step

        bicoherence = np.real(np.abs(bicoherence))

        if method in ['sigl', 'nagashima']:
            normalization = np.sqrt(normalization)

        if method != 'bispectrum':
            bicoherence /= normalization
        else:
            bicoherence = np.log(bicoherence)

        if count == 0:
            raise IndexError(
                'bicoherence: first block needs %d samples but sigs has shape '
                '%s' % (self.blklen, sigs.shape))

        self.bicoherence = bicoherence
        return bicoherence

    def plot(self, fig=None, ax=None):

        if fig is None:
            fig = plt.figure()
        if ax is None:
            ax = fig.gca()

        fmax = self.fs / 2.0
        bicoherence = np.copy(self.bicoherence)
        n_freq = bicoherence.shape[0]
        np.flipud(bicoherence)[np.triu_indices(n_freq, 1)] = 0
        bicoherence[np.triu_indices(n_freq, 1)] = 0
        bicoherence = bicoherence[:, :n_freq // 2 + 1]

        vmin, vmax = compute_vmin_vmax(bicoherence, tick=1e-15, percentile=1)
        cax = ax.imshow(
            bicoherence, cmap=plt.cm.viridis, aspect='auto',
            vmin=vmin, vmax=vmax,
            origin='lower', extent=(0, fmax // 2, 0, fmax),
            interpolation='none')
        ax.set_title('Bicoherence (%s)' % self.method)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        add_colorbar(fig, cax, vmin, vmax, unit='', ax=ax)
        return fig

    def main_frequency(self):
        pass


def phase_amplitude(signals, phase=True, amplitude=True):
    """Extract instantaneous phase and amplitude with Hilbert transform"""
    # one dimension array
    if signals.ndim == 1:
        signals = signals[None, :]
        one_dim = True
    elif signals.ndim == 2:
        one_dim = False
    else:
        raise ValueError('Impoosible to compute phase_amplitude with ndim ='
                         ' %s.' % (signals.ndim, ))

    n_epochs, n_points = signals.shape
    n_fft = compute_n_fft(signals)

    sig_phase = np.empty(signals.shape) if phase else None
    sig_amplitude = np.empty(signals.shape) if amplitude else None
    for i, sig in enumerate(signals):
        sig_complex = hilbert(sig, n_fft)

        if phase:
            sig_phase[i] = np.angle(sig_complex)
        if amplitude:
            sig_amplitude[i] = np.abs(sig_complex)

    # go back to the initial numebr of time points
    sig_phase = sig_phase[:, :n_points]
    sig_amplitude = sig_amplitude[:, :n_points]

    # one dimension array
    if one_dim:
        if phase:
            sig_phase = sig_phase[0]
        if amplitude:
            sig_amplitude = sig_amplitude[0]

    return sig_phase, sig_amplitude


def square(c):
    """square a complex"""
    return np.real(c * np.conjugate(c))


def crop_for_fast_hilbert(signals):
    """
    Crop the signal to have a good prime decomposition, for hilbert filter.
    """
    if signals.ndim < 2:
        tmax = signals.shape[0]
        while prime_factors(tmax)[-1] > 20:
            tmax -= 1
        return signals[:tmax]
    else:
        tmax = signals.shape[1]
        while prime_factors(tmax)[-1] > 20:
            tmax -= 1
        return signals[:, :tmax]


def compute_n_fft(signals):
    """
    Compute the number of element in the FFT to have a good prime
    decomposition, for hilbert filter.
    """
    n_fft = signals.shape[-1]
    while prime_factors(n_fft)[-1] > 20:
        n_fft += 1
    return n_fft


def prime_factors(n):
    """
    Decomposition in prime factor.
    Used to find signal length that speed up Hilbert transform.
    """
    assert n > 0
    assert isinstance(n, int)

    decomposition = []
    for i in chain([2], range(3, int(np.sqrt(n)) + 1, 2)):
        while n % i == 0:
            n = n // i
            decomposition.append(i)
        if n == 1:
            break
    if n != 1:
        decomposition.append(n)
    return decomposition


def compute_vmin_vmax(spec, vmin=None, vmax=None, tick=0.01, percentile=1):
    """compute automatic scale for plotting `spec`"""
    if percentile > 100:
        percentile = 100
    if percentile < 0:
        percentile = 0

    if vmin is None:
        vmin = np.floor(np.percentile(spec, percentile) / tick) * tick
    if vmax is None:
        vmax = np.ceil(np.percentile(spec, 100 - percentile) / tick) * tick

    return vmin, vmax


def add_colorbar(fig, cax, vmin, vmax, unit='', ax=None):
    """Add a colorbar only once in a multiple subplot figure"""
    if vmin == vmax:
        vmin = 0.
    log_scale = np.floor(np.log10((vmax - vmin) / 3.0))
    scale = 10. ** log_scale
    vmax = np.floor(vmax / scale) * scale
    vmin = np.ceil(vmin / scale) * scale

    range_scale = int((vmax - vmin) / scale)
    step = np.ceil(range_scale / 7.) * scale

    ticks_str = '%%.%df' % max(0, -int(log_scale))
    ticks = np.arange(vmin, vmax + step, step)
    ticks_str += ' ' + unit

    fig.subplots_adjust(right=0.85)
    if ax is None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.03, 0.8])
    else:
        cbar_ax = None
    cbar = fig.colorbar(cax, ax=ax, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels([ticks_str % t for t in ticks])
