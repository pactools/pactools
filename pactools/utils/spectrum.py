import numpy as np
import scipy as sp
from scipy.linalg import hankel
from scipy.signal import hilbert
import matplotlib.pyplot as plt

from .maths import square, is_power2, prime_factors, compute_n_fft
from .viz import compute_vmin_vmax


class Spectrum(object):
    def __init__(self, block_length=1024, fft_length=None, step=None,
                 wfunc=np.hamming, fs=1., donorm=True):
        """
        Spectral estimator

        Parameters
        ----------
        block_length : int
            Length of each signal block, on which we estimate the spectrum

        fft_length : int or None
            Length of FFT, should be greater or equal to block_length.
            If None, it is set to block_length

        step : int or None
            Step between successive blocks
            If None, it is set to half the block length (i.e. 0.5 overlap)

        wfunc : function
            Function used to compute the weitghting window on each block.
            Examples: np.ones, np.hamming, np.bartlett, np.blackman, ...

        fs : float
            Sampling frequency

        donorm : boolean
            If True, the amplitude is normalized

        """
        self.block_length = block_length
        self.fft_length = fft_length
        self.step = step
        self.wfunc = wfunc
        self.fs = fs
        self.donorm = donorm
        self.psd = []

    def check_params(self):
        # block_length
        if self.block_length < 0:
            raise ValueError('Block length is negative')
        self.block_length = int(self.block_length)

        # fft_length
        if self.fft_length is None:
            fft_length = int(2 ** np.ceil(np.log2(self.block_length)))
        else:
            fft_length = int(self.fft_length)
        if not is_power2(fft_length):
            raise ValueError('FFT length should be a power of 2')
        if fft_length < self.block_length:
            raise ValueError('Block length is greater than FFT length')

        # step
        if self.step is None:
            step = int(self.block_length // 2)
        else:
            step = int(self.step)
        if step <= 0 or step > self.block_length:
            raise ValueError('Invalid step between blocks: %s' % (step, ))

        return fft_length, step

    def periodogram(self, signals, hold=False):
        """
        Computes the estimation (in dB) for each epoch in a signal

        Parameters
        ----------
        signals : array, shape (n_epochs, n_points)
            Signals from which one computes the power spectrum

        hold : boolean, default = False
            If True, the estimation is appended to the list of previous
            estimations, else, the list is emptied and only the current
            estimation is stored.

        Returns
        -------
        psd : array, shape (n_epochs, n_freq)
            Power spectrum estimated with a Welsh method on each epoch
            n_freq = fft_length // 2 + 1
        """
        fft_length, step = self.check_params()

        signals = np.atleast_2d(signals)

        window = self.wfunc(self.block_length)
        n_epochs, tmax = signals.shape
        n_freq = fft_length // 2 + 1

        psd = np.zeros((n_epochs, n_freq))

        for i, sig in enumerate(signals):
            block = np.arange(self.block_length)

            # iterate on blocks
            count = 0
            while block[-1] < sig.size:
                psd[i] += np.abs(sp.fft(window * sig[block], fft_length,
                                        0))[:n_freq] ** 2
                count = count + 1
                block = block + step
            if count == 0:
                raise IndexError(
                    'spectrum: first block has %d samples but sig has %d '
                    'samples' % (block[-1] + 1, sig.size))

            # normalize
            if self.donorm:
                scale = 1.0 / (count * (np.sum(window) ** 2))
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
        fft_length, _ = self.check_params()
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
        freq = np.linspace(0, fmax, fft_length // 2 + 1)

        for label_, color, psd in zip(labels, colors, self.psd):
            psd = 10.0 * np.log10(np.maximum(psd, 1.0e-16))
            for i in range(replicate + 1):
                label = label_ if i == 0 else ''
                fig.plot(freq + i * fmax, psd.T[::(-1) ** i], label=label,
                         color=color)

        fig.grid(True)
        fig.set_title(title)
        fig.set_xlabel('Frequency (Hz)')
        fig.set_ylabel('Amplitude (dB)')
        if plot_legend:
            fig.legend(loc=0)
        return fig

    def main_frequency(self):
        """Extract the frequency of the maximum in the spectrum"""
        fft_length, _ = self.check_params()
        n_freq = fft_length // 2 + 1
        freq = np.linspace(0, self.fs / 2, n_freq)
        psd = self.psd[-1][0, :]
        return freq[np.argmax(psd[1:]) + 1]


class Coherence(Spectrum):
    def __init__(self, block_length=1024, fft_length=None, step=None,
                 wfunc=np.hamming, fs=1.):
        """
        Coherence estimator

        Parameters
        ----------
        block_length : int
            Length of each signal block, on which we estimate the spectrum

        fft_length : int or None
            Length of FFT, should be greater or equal to block_length.
            If None, it is set to block_length

        step : int or None
            Step between successive blocks
            If None, it is set to half the block length (i.e. 0.5 overlap)

        wfunc : function
            Function used to compute the weitghting window on each block.
            Examples: np.ones, np.hamming, np.bartlett, np.blackman, ...

        fs : float
            Sampling frequency

        donorm : boolean
            If True, the amplitude is normalized
        """
        super(Coherence, self).__init__(block_length=block_length,
                                        fft_length=fft_length, step=step,
                                        wfunc=wfunc, fs=fs)

        self.coherence = None

    def fit(self, sigs_a, sigs_b):
        """
        Computes the coherence for two signals.
        It is symmetrical, and slightly faster if n_signals_a < n_signals_b.


        Parameters
        ----------
        sigs_a : array, shape (n_signals_a, n_epochs, n_points)
            Signal from which one computes the coherence

        sigs_b : array, shape (n_signals_b, n_epochs, n_points)
            Signal from which one computes the coherence

        Returns
        -------
        coherence : array, shape (n_signals_a, n_signals_b, n_freqs)
            Complex coherence of sigs_a and sigs_b over all epochs.
            n_freqs = fft_length // 2 + 1
        """
        fft_length, step = self.check_params()

        n_signals_a, n_epochs, n_points = sigs_a.shape
        n_signals_b, n_epochs, n_points = sigs_b.shape
        if sigs_a.shape[1:] != sigs_b.shape[1:]:
            raise ValueError('Incompatible shapes: %s and %s' %
                             (sigs_a.shape[1:], sigs_b.shape[1:]))

        window = self.wfunc(self.block_length)
        n_freq = fft_length // 2 + 1
        coherence = np.zeros((n_signals_a, n_signals_b, n_freq),
                             dtype=np.complex128)
        norm_a = np.zeros((n_signals_a, n_freq), dtype=np.float64)
        norm_b = np.zeros((n_signals_b, n_freq), dtype=np.float64)

        # iterate on blocks
        count = 0
        for i_epoch in range(n_epochs):
            block = np.arange(self.block_length)
            while block[-1] < n_points:

                for i_a in range(n_signals_a):
                    F_a = sp.fft(window * sigs_a[i_a, i_epoch, block],
                                 fft_length, 0)[:n_freq]
                    norm_a[i_a] += square(F_a)

                    for i_b in range(n_signals_b):
                        F_b = sp.fft(window * sigs_b[i_b, i_epoch, block],
                                     fft_length, 0)[:n_freq]
                        # compute only once
                        if i_a == 0:
                            norm_b[i_b] += square(F_b)

                        coherence[i_a, i_b] += F_a * np.conjugate(F_b)

                count = count + 1
                block = block + step

        normalization = np.sqrt(norm_a[:, None, :] * norm_b[None, :, :])
        coherence /= normalization

        if count == 0:
            raise IndexError(
                'bicoherence: first block needs %d samples but sigs has shape '
                '%s' % (self.block_length, sigs_a.shape))

        self.coherence = coherence
        return self.coherence

    def plot(self, fig=None, ax=None):
        """Not Implemented"""
        return fig

    def main_frequency(self):
        pass


class Bicoherence(Spectrum):
    def __init__(self, block_length=1024, fft_length=None, step=None,
                 wfunc=np.hamming, fs=1.):
        """
        Bicoherence estimator

        Parameters
        ----------
        block_length : int
            Length of each signal block, on which we estimate the spectrum

        fft_length : int or None
            Length of FFT, should be greater or equal to block_length.
            If None, it is set to block_length

        step : int or None
            Step between successive blocks
            If None, it is set to half the block length (i.e. 0.5 overlap)

        wfunc : function
            Function used to compute the weitghting window on each block.
            Examples: np.ones, np.hamming, np.bartlett, np.blackman, ...

        fs : float
            Sampling frequency

        donorm : boolean
            If True, the amplitude is normalized
        """
        super(Bicoherence, self).__init__(block_length=block_length,
                                          fft_length=fft_length, step=step,
                                          wfunc=wfunc, fs=fs)

    def fit(self, sigs, method='hagihira'):
        """
        Computes the bicoherence for one signal

        Parameters
        ----------
        sigs : array, shape (n_epochs, n_points)
            Signal from which one computes the bicoherence

        method : string in ('hagihira', 'sigl', 'nagashima', 'bispectrum')
            Normalization used for the bicoherence

        Returns
        -------
        bicoherence : array, shape (n_freq, n_freq)
            Complex bicoherence computed on the input signal.
            n_freq = fft_length // 2 + 1
        """
        fft_length, step = self.check_params()
        self.method = method

        sigs = np.atleast_2d(sigs)
        n_epochs, n_points = sigs.shape

        window = self.wfunc(self.block_length)
        n_freq = fft_length // 2 + 1
        bicoherence = np.zeros((n_freq, n_freq), dtype=np.complex128)
        normalization = np.zeros((n_freq, n_freq), dtype=np.float64)

        # iterate on blocks
        count = 0
        for i_epoch in range(n_epochs):
            block = np.arange(self.block_length)
            while block[-1] < n_points:
                F = sp.fft(window * sigs[i_epoch, block], fft_length,
                           0)[:n_freq]
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
                    raise (ValueError("Method '%s' unkown." % method))

                count = count + 1
                block = block + step

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
                '%s' % (self.block_length, sigs.shape))

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
        ax.imshow(bicoherence, cmap=plt.cm.viridis, aspect='auto', vmin=vmin,
                  vmax=vmax, origin='lower', extent=(0, fmax // 2, 0, fmax),
                  interpolation='none')
        ax.set_title('Bicoherence (%s)' % self.method)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency (Hz)')
        # add_colorbar(fig, cax, vmin, vmax, unit='', ax=ax)
        return ax

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
        raise ValueError('Impossible to compute phase_amplitude with ndim ='
                         ' %s.' % (signals.ndim, ))

    n_epochs, n_points = signals.shape
    n_fft = compute_n_fft(signals)

    sig_phase = np.empty(signals.shape) if phase else None
    sig_amplitude = np.empty(signals.shape) if amplitude else None
    for i, sig in enumerate(signals):
        sig_complex = hilbert(sig, n_fft)[:n_points]

        if phase:
            sig_phase[i] = np.angle(sig_complex)
        if amplitude:
            sig_amplitude[i] = np.abs(sig_complex)

    # one dimension array
    if one_dim:
        if phase:
            sig_phase = sig_phase[0]
        if amplitude:
            sig_amplitude = sig_amplitude[0]

    return sig_phase, sig_amplitude


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
