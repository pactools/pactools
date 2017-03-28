import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from .spectrum import Spectrum
from .progress_bar import ProgressBar


def dehummer(sig, fs, enf=50.0, hmax=5, block_length=2048, draw='',
             progress_bar=True):
    """Removes the ENF signal and its harmonics

    sig    : input 1D signal
    fs     : sampling frequency
    enf    : electrical network frequency
    hmax   : maximum number of harmonics
    block_length : length of FFTs
    draw   : list of plots

    returns the denoised signal
    """
    hmax = min(hmax, int(0.5 * fs / enf))
    if sig.ndim > 2 or (sig.ndim == 2 and min(sig.shape) > 1):
        raise ValueError('Input signal should be 1D. Got %s.' % (sig.shape, ))
    input_shape = sig.shape
    sig = np.ravel(sig)

    block_length = min(block_length, sig.size)
    block_length_o2 = block_length // 2

    # -------- the window and its shift by block_length/2 must sum to 1.0
    window = np.hamming(block_length)
    window[0:block_length_o2] /= (
        window[0:block_length_o2] + window[block_length_o2:block_length])
    window[block_length_o2:block_length] = np.flipud(window[0:block_length_o2])

    if hmax == 0:
        return sig
    result = np.zeros_like(sig)

    # -------- prepare an array with estimated frequencies
    tmax = len(sig)
    freq = np.zeros(2 + 2 * tmax // block_length)
    kf = 0
    if progress_bar:
        bar = ProgressBar(max_value=len(freq), title='dehumming %.0f Hz' % enf)

    # -------- process successive blocks
    for tmid in range(0, tmax + block_length_o2, block_length_o2):
        # -------- initial and final blocks are truncated
        tstart = tmid - block_length_o2
        if tstart < 0:
            wstart = -tstart
            tstart = 0
        else:
            wstart = 0
        tstop = tmid + block_length_o2
        if tstop > tmax:
            wstop = block_length + tmax - tstop
            tstop = tmax
        else:
            wstop = block_length

        # -------- search for the frequency
        f0 = enf
        sigenf = single_estimate(sig[tstart:tstop], f0, fs, hmax)
        best_f = f0
        best_sigout = sig[tstart:tstop] - sigenf
        best_energy = np.dot(best_sigout.T, best_sigout)
        for delta in {0.1, 0.01}:
            shift_max = 9 if delta == 0.01 else 9
            shifts = np.arange(1, shift_max + 1)
            shifts = np.r_[-shifts[::-1], shifts]

            for shift in shifts:
                f = f0 + shift * delta
                sigenf = single_estimate(sig[tstart:tstop], f, fs, hmax)
                sigout = sig[tstart:tstop] - sigenf
                energy = np.dot(sigout.T, sigout)
                # we keep the frequency f that removes the most energy
                if energy < best_energy:
                    best_f = f
                    best_sigout = sigout
                    best_energy = energy
            f0 = best_f

        # -------- this block has been processed, save it
        result[tstart:tstop] += best_sigout * window[wstart:wstop]

        freq[kf] = best_f
        kf += 1
        if progress_bar and kf % 10 == 0:
            bar.update(kf)
    if progress_bar:
        bar.update(bar.max_value)

    # -------- plot estimated electrical network frequency
    if 'f' in draw or 'z' in draw:
        t = np.linspace(0, tmax / fs, len(freq))
        plt.figure('Estimated electrical network frequency')
        plt.title('Estimated electrical network frequency')

        plt.plot(t, freq / enf, label='%.0fHz' % enf)
        plt.ylabel('ENF fluctuation'),
        plt.xlabel('Time (sec)')
        plt.legend(loc=0)

    # -------- plot long term spectum of noisy and denoised signals
    if 'd' in draw or 'z' in draw:
        sp = Spectrum(block_length=block_length, fs=fs, donorm=True,
                      wfunc=np.blackman)
        sp.periodogram(sig)
        sp.periodogram(result, hold=True)
        sp.plot('Power spectral density before/after dehumming', fscale='lin')

    return result.reshape(input_shape)


def single_estimate(sigin, f, fs, hmax):
    """Estimate the contribution of electrical network
    signal in sigin, if ENF is f.

    return the estimated signal
    """
    X = np.empty((len(sigin), hmax))
    fact = 2.0 * np.pi * f / fs * np.arange(1, hmax + 1)[:, None]
    p = np.arange(len(sigin))[:, None]
    X = np.dot(fact, p.T)
    X = np.concatenate([X, X + np.pi / 2]).T
    X = np.cos(X)
    XX = np.dot(X.T, X)
    Xy = np.dot(X.T, sigin)
    theta = linalg.solve(XX, Xy)
    return np.dot(X, theta)
