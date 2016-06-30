import numpy as np
import matplotlib.pyplot as plt

from mne.preprocessing.peak_finder import peak_finder

from .utils.spectrum import crop_for_fast_hilbert
from .plot_comodulogram import add_colorbar
from .modulation_index import multiple_band_pass


def theta_trough_lock(sig, fs,
                      low_fq=6.0,
                      low_fq_width=2.0,
                      high_fq_range=np.linspace(2.0, 160.0, 80),
                      high_fq_width=8.0,
                      t_plot=1.0,
                      method=1,
                      save_name=None):
    """
    Plot the theta-trough locked Time-frequency plot of mean power
    modulation time-locked to the theta trough

    Parameters
    ----------
    sig           : one dimension signal
    fs            : sampling frequency
    low_fq        : low frequency(phase signal)
    low_fq_width  : width of the band-pass filter
    high_fq_range : high frequency range to compute the MI (amplitude signal)
    high_fq_width : width of the band-pass filter
    t_plot        : in second, time to plot around the troughs
    method        : in {0, 1}, choose band pass filter method
                    (in multiple_band_pass)

    """
    n_cycles = None
    low_fq = np.atleast_1d(low_fq)

    sig = sig.astype(np.float64).ravel()
    sig = crop_for_fast_hilbert(sig)

    # compute a number of band-pass filtered and Hilbert filtered signals
    sigdriv_complex = multiple_band_pass(sig, fs, low_fq, low_fq_width,
                                         method=method)[0]
    sigdriv = np.real(sigdriv_complex)

    if False:
        # find the trough in the sigdriv with a peak finder
        thresh = (sigdriv.max() - sigdriv.min()) / 10.
        trough_loc, trough_mag = peak_finder(sigdriv, thresh=thresh,
                                             extrema=-1)
    else:
        # find the trough in the phase
        phase = np.angle(sigdriv_complex)
        trough_loc, _ = peak_finder(phase, extrema=1)
        trough_mag = sigdriv[trough_loc]

    # plot the sigdriv troughs
    if True:
        n_points = 3000
        t = np.arange(n_points) / float(fs)
        plt.figure(figsize=(16, 5))
        plt.plot(t, sig[:n_points], label='signal')
        plt.plot(t, sigdriv[:n_points], label='driver')
        plt.plot(trough_loc[trough_loc < n_points] / float(fs),
                 trough_mag[trough_loc < n_points], 'o', label='trough')
        plt.xlabel('Time (sec)')
        plt.title("Driver's trough detection")
        plt.legend(loc=0)

    # extract several signals with band-pass filters
    filtered_high = multiple_band_pass(sig, fs, high_fq_range, high_fq_width,
                                       n_cycles=n_cycles, method=method)

    # normalize each signal independently
    filtered_high -= filtered_high.mean(axis=1)[:, None]
    filtered_high /= filtered_high.std(axis=1)[:, None]
    # get the power (np.abs(filtered_high) ** 2)
    filtered_high *= np.conj(filtered_high)
    filtered_high = np.real(filtered_high)

    # subtract the mean power
    filtered_high -= filtered_high.mean(axis=1)[:, None]

    mask = (trough_mag >= 0)

    trough_loc_masked = trough_loc[mask]

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs = axs.ravel()

    # plot the lower part
    plot_trough_locked_time(
        sig, fs, trough_loc=trough_loc_masked,
        t_plot=t_plot, fig=fig, ax=axs[1], ylim=None)

    # plot the higher part
    plot_trough_locked_time_frequency(
        filtered_high, fs, high_fq_range,
        trough_loc=trough_loc_masked, t_plot=t_plot, fig=fig, ax=axs[0],
        vmin=None, vmax=None)

    # save the figure
    if save_name is None:
        save_name = ('%s_wlo%.2f_whi%.1f'
                     % (method, low_fq_width, high_fq_width))
    fig.savefig(save_name + '.png')


def trough_locked_percentile(signals, fs, trough_loc, t_plot,
                             percentiles=['mean']):
    """
    Compute the mean of each signal in signals, locked to the troughs

    Parameters
    ----------
    signals    : shape (n_signals, tmax)
    fs         : sampling frequency
    trough_loc : indices of the trough locations
    t_plot     : in second, time to plot around the troughs
    percentiles: list of precentile to compute. It can also include c,
                 'std' or 'ste' (mean, standard deviation or standard error).

    Returns
    -------
    evoked_signals :
    """
    n_signals, tmax = signals.shape
    n_points = int(fs * t_plot / 2.) * 2 + 1
    n_percentiles = len(percentiles)

    # remove the troughs to close to the start or the end
    mask = np.logical_and(trough_loc > n_points,
                          trough_loc < tmax - n_points)

    # build indices matrix: each line is a index range around trough locations
    indices = np.tile(trough_loc[mask], (n_points, 1)).T
    indices = indices + np.arange(n_points) - n_points // 2

    n_troughs = indices.shape[0]

    # compute the evoked signals (trough-locked mean)
    evoked_signals = np.zeros((n_signals, n_percentiles, n_points))
    for i_s in range(n_signals):
        for i_p, p in enumerate(percentiles):
            if isinstance(p, int):
                evoked_signals[i_s, i_p] = np.percentile(
                    signals[i_s][indices], p, axis=0)
                continue

            mean = np.mean(signals[i_s][indices], axis=0)
            if p == 'mean':
                evoked_signals[i_s, i_p] = mean
            else:
                std = np.std(signals[i_s][indices], axis=0)
                if p == 'std+':
                    evoked_signals[i_s, i_p] = mean + std
                elif p == 'std-':
                    evoked_signals[i_s, i_p] = mean - std
                elif p == 'ste+':
                    evoked_signals[i_s, i_p] = mean + std / np.sqrt(n_troughs)
                elif p == 'ste-':
                    evoked_signals[i_s, i_p] = mean - std / np.sqrt(n_troughs)
                else:
                    raise(ValueError, 'wrong percentile string: %s' % p)

    return evoked_signals


def plot_trough_locked_time_frequency(filtered_high, fs, high_fq_range,
                                      trough_loc, t_plot, fig=None, ax=None,
                                      vmin=None, vmax=None):
    """
    Plot the theta-trough locked Time-frequency
    """
    # compute the evoked signals (trough-locked mean)
    evoked_signals = trough_locked_percentile(filtered_high, fs,
                                              trough_loc, t_plot)
    evoked_signals = evoked_signals[:, 0, :]

    if fig is None:
        fig = plt.figure()
        ax = fig.gca()

    # plot
    vmax = evoked_signals.max() if vmax is None else vmax
    vmin = -vmax
    cax = ax.imshow(evoked_signals,
                    cmap=plt.get_cmap('RdBu_r'),
                    vmin=vmin, vmax=vmax,
                    aspect='auto',
                    origin='lower',
                    interpolation='none',
                    extent=(-t_plot / 2, t_plot / 2,
                            high_fq_range[0], high_fq_range[-1]))

    # ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Driver trough-locked Time-frequency decomposition')

    # plot the colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    add_colorbar(fig, cax, vmin, vmax, unit='', ax=None)

    return vmin, vmax


def plot_trough_locked_time(sig, fs, trough_loc, t_plot,
                            fig=None, ax=None, ylim=None):
    """
    Compute and plot the trough-locked mean of the signal sig
    """
    # percentiles = [5, 'mean', 95]
    # percentiles = ['mean']
    percentiles = ['std+', 'mean', 'std-']
    lines = 'gbg'

    labels = {'std+': r'$\mu+\sigma$',
              'std-': r'$\mu-\sigma$',
              'ste+': r'$\mu+\sigma/\sqrt{n}$',
              'ste-': r'$\mu-\sigma/\sqrt{n}$',
              'mean': r'$\mu$',
              }

    # compute the evoked signals (trough-locked mean)
    evoked_sig = trough_locked_percentile(sig[None, :], fs,
                                          trough_loc, t_plot,
                                          percentiles)
    n_signals, n_percentiles, n_points = evoked_sig.shape

    if fig is None:
        fig = plt.figure()
        ax = fig.gca()

    # plot
    t = (np.arange(n_points) - n_points // 2) / float(fs)
    for i, p in enumerate(percentiles):
        label = ('%d %%' % p) if isinstance(p, int) else labels[p]
        ax.plot(t, evoked_sig[0, i], lines[i], label=label)

    ax.set_xlabel('Time (sec)')
    ax.set_title('Driver trough-locked average of raw signal')
    ax.legend(loc='lower center', ncol=5, labelspacing=0.)
    ax.grid('on')

    # make room for the legend or apply specified ylim
    if ylim is None:
        ylim = ax.get_ylim()
        ylim = (ylim[0] - (ylim[1] - ylim[0]) * 0.2, ylim[1])
    ax.set_ylim(ylim)

    return ylim
