import numpy as np
import matplotlib.pyplot as plt

from .plot_comodulogram import add_colorbar


def plot_trough_locked_time_frequency(filtered_high, fs, high_fq_range,
                                      trough_loc, t_plot, mask=None, fig=None,
                                      ax=None, vmin=None, vmax=None):
    """
    Plot the theta-trough locked Time-frequency
    """
    # normalize each signal independently
    n_frequencies, n_epochs, n_points = filtered_high.shape

    # normalization is done everywhere, but mean is computed
    #  only where mask == 1
    if mask is not None:
        masked_filtered_high = filtered_high[:, mask == 0]
    else:
        masked_filtered_high = filtered_high.reshape(n_frequencies, -1)

    mean = masked_filtered_high.mean(axis=1)[:, None, None]
    std = masked_filtered_high.std(axis=1)[:, None, None]

    filtered_high -= mean
    filtered_high /= std

    # get the power (np.abs(filtered_high) ** 2)
    filtered_high *= np.conj(filtered_high)
    filtered_high = np.real(filtered_high)

    # subtract the mean power.
    if mask is not None:
        masked_filtered_high = filtered_high[:, mask == 0]
    else:
        masked_filtered_high = filtered_high.reshape(n_frequencies, -1)
    mean = masked_filtered_high.mean(axis=1)[:, None, None]
    filtered_high -= mean

    # compute the evoked signals (trough-locked mean)
    evoked_signals = trough_locked_percentile(filtered_high, fs, trough_loc,
                                              t_plot)
    evoked_signals = evoked_signals[:, 0, :]

    if fig is None:
        fig = plt.figure()
        ax = fig.gca()

    # plot
    vmax = np.abs(evoked_signals).max() if vmax is None else vmax
    vmin = -vmax
    cax = ax.imshow(
        evoked_signals, cmap=plt.get_cmap('RdBu_r'), vmin=vmin, vmax=vmax,
        aspect='auto', origin='lower', interpolation='none', extent=(
            -t_plot / 2, t_plot / 2, high_fq_range[0], high_fq_range[-1]))

    # ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Driver trough-locked Time-frequency decomposition')

    # plot the colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    add_colorbar(fig, cax, vmin, vmax, unit='', ax=None)

    return vmin, vmax


def plot_trough_locked_time(sig, fs, trough_loc, t_plot, fig=None, ax=None,
                            ylim=None):
    """
    Compute and plot the trough-locked mean of the signal sig
    """
    # percentiles = [5, 'mean', 95]
    # percentiles = ['mean']
    percentiles = ['std+', 'mean', 'std-']
    lines = ['k--', 'k-', 'k--']  # 'gbg'

    labels = {
        'std+': r'$\mu+\sigma$',
        'std-': r'$\mu-\sigma$',
        'ste+': r'$\mu+\sigma/\sqrt{n}$',
        'ste-': r'$\mu-\sigma/\sqrt{n}$',
        'mean': r'$\mu$',
    }

    # compute the evoked signals (trough-locked mean)
    evoked_sig = trough_locked_percentile(sig[None, :], fs, trough_loc, t_plot,
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
    ax.set_xlim([t[0], t[-1]])

    return ylim


def trough_locked_percentile(signals, fs, trough_loc, t_plot,
                             percentiles=['mean']):
    """
    Compute the mean of each signal in signals, locked to the troughs

    Parameters
    ----------
    signals    : shape (n_signals, n_epochs, n_points)
    fs         : sampling frequency
    trough_loc : indices of the trough locations
    t_plot     : in second, time to plot around the troughs
    percentiles: list of precentile to compute. It can also include c,
                 'std' or 'ste' (mean, standard deviation or standard error).

    Returns
    -------
    evoked_signals :
    """
    n_signals, n_epochs, n_points = signals.shape
    n_window = int(fs * t_plot / 2.) * 2 + 1
    n_percentiles = len(percentiles)

    # build indices matrix: each line is a index range around trough locations
    indices = np.tile(trough_loc, (n_window, 1)).T
    indices = indices + np.arange(n_window) - n_window // 2

    # ravel the epochs since we now have isolated events
    signals = signals.reshape(n_signals, -1)

    n_troughs = indices.shape[0]
    assert n_troughs > 0

    # compute the evoked signals (trough-locked mean)
    evoked_signals = np.zeros((n_signals, n_percentiles, n_window))
    for i_s in range(n_signals):
        for i_p, p in enumerate(percentiles):
            if isinstance(p, int):
                evoked_signals[i_s, i_p] = np.percentile(signals[i_s][indices],
                                                         p, axis=0)
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
                    raise (ValueError, 'wrong percentile string: %s' % p)

    return evoked_signals
