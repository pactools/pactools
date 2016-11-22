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
    spec, xlim, sigdriv, sigdriv_imag = model.amplitude_frequency(
        mode=mode, xlim=xlim)

    if fig is None:
        fig = plt.figure(title)
    try:
        ax = fig.gca()
    except:
        ax = fig
    # ax.cla()

    # -------- plot spectrum
    colors = 'bcgyrm'
    n_frequency, n_amplitude = spec.shape
    if sigdriv_imag is None:
        max_i = (n_amplitude - 1)
    else:
        max_i = (n_amplitude - 1) / 2
    ploted_amplitudes = np.linspace(0, max_i, 5)
    ploted_amplitudes = np.floor(ploted_amplitudes).astype(np.int)

    frequencies = np.linspace(frange[0], frange[1], n_frequency)
    for i, i_amplitude in enumerate(ploted_amplitudes[::-1]):

        # build label
        # if sigdriv_imag is not None:
        #     str_x_imag = r' \bar{x}=%.2f' % sigdriv_imag[i_amplitude]
        # else:
        str_x_imag = ''
        label = r'$x=%.2f%s$' % (sigdriv[i_amplitude], str_x_imag)

        ax.plot(frequencies, spec[:, i_amplitude], color=colors[i],
                label=label)

    # plot simple periodogram
    if True:
        spect = Spectrum(block_length=128, fs=model.fs, wfunc=np.blackman)
        sigin = model.sigin
        if model.mask is not None:
            sigin = sigin[model.mask != 0]
        spect.periodogram(sigin)
        fft_length, _ = spect.check_params()
        n_frequency = fft_length // 2 + 1
        psd = spect.psd[0][0, 0:n_frequency]
        psd = 10.0 * np.log10(np.maximum(psd, 1.0e-12))
        psd += -psd.mean() + 20.0 * np.log10(sigin.std())
        frequencies = np.linspace(0, spect.fs / 2, n_frequency)
        ax.plot(frequencies, psd, '--k', label='PSD')

    if title == '':
        title = model.get_title(name=True, bic=True)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (dB)')
    ax.legend(loc=0)
    vmin, vmax = compute_vmin_vmax(spec, vmin, vmax, 0.1, percentile=0)
    vmin_, vmax_ = compute_vmin_vmax(psd, None, None, 0.1, percentile=0)
    vmin = min(vmin, vmin_)
    vmax = max(vmax, vmax_)
    ax.set_ylim((vmin, vmax))
    ax.grid('on')

    return fig
