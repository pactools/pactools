from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from ..utils.spectrum import Spectrum


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


def plot_dar_lines(model, title='', frange=None, mode='',
                   fig=None, xlim=None, vmin=None, vmax=None):
    """
    Represent the power spectral density as a function of
    the amplitude of the driving signal

    model   : the AR model to plot
    title   : title of the plot
    frange  : frequency range
    mode    : normalisation mode ('c' = centered, 'v' = unit variance)
    fig     : reuse this figure if defined
    xlim    : force xlim to these values if defined
    vmin    : minimum power to plot (dB)
    vmax    : maximum power to plot (dB)
    """
    if frange is None:
        frange = [0, model.fs / 2.0]

    # -------- get amplitude-frequency spectrum from model
    spec, xlim = model.amplitude_frequency(frange=frange, mode=mode, xlim=xlim)

    if fig is None:
        fig = plt.figure(title)
    try:
        ax = fig.gca()
    except:
        ax = fig
    ax.cla()

    # -------- plot spectrum
    colors = 'bcgyrm'
    n_frequency, n_amplitude = spec.shape
    amplitudes = np.linspace(xlim[0], xlim[1], n_amplitude)
    ploted_amplitudes = np.floor(np.linspace(0, n_amplitude - 1, 5))
    ploted_amplitudes = ploted_amplitudes.astype(np.int)

    frequencies = np.linspace(frange[0], frange[1], n_frequency)
    for i, i_amplitude in enumerate(ploted_amplitudes[::-1]):
        ax.plot(frequencies, spec[:, i_amplitude], color=colors[i],
                label=r'$x=%.2f$' % amplitudes[i_amplitude])

    # plot simple periodogram
    if True:
        spect = Spectrum(blklen=128, fs=model.fs, wfunc=np.blackman)
        spect.periodogram(model.sigin)
        n_frequency = spect.fftlen // 2 + 1
        psd = spect.psd[0][0, 0:n_frequency]
        psd = 10.0 * np.log10(np.maximum(psd, 1.0e-16))
        psd += -psd.mean() + 20.0 * np.log10(model.sigin.std())
        frequencies = np.linspace(0, spect.fs / 2, n_frequency)
        ax.plot(frequencies, psd, '--k', label='PSD')

    if title == '':
        title = model.get_title(name=True, logl=True)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.legend(loc=0)

    ylim = ax.get_ylim()
    if vmin is not None:
        ylim = (vmin, ylim[1])
    if vmax is not None:
        ylim = (ylim[0], vmax)
    ax.set_ylim(ylim)
    ax.grid('on')

    return fig
