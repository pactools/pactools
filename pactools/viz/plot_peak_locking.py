import numpy as np
import matplotlib.pyplot as plt

from .utils import add_colorbar


def plot_peak_locking(
        fs, evoked_time_frequency, evoked_time, t_plot, high_fq_range,
        percentiles, axs=None, vmin=None, vmax=None, ylim=None):
    """
    vmin, vmax:
            Min and max value for the colorbar.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure instance containing the plot.
    """
    if axs is None:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
        axs = axs.ravel()
    else:
        fig = axs[0].figure

    # plot the trough-locked time-frequency
    ax = axs[0]
    vmax = np.abs(evoked_time_frequency).max() if vmax is None else vmax
    vmin = -vmax
    cax = ax.imshow(evoked_time_frequency, cmap=plt.get_cmap('RdBu_r'),
                    vmin=vmin, vmax=vmax, aspect='auto', origin='lower',
                    interpolation='none', extent=(-t_plot / 2, t_plot / 2,
                                                  high_fq_range[0],
                                                  high_fq_range[-1]))

    # ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Driver trough-locked Time-frequency decomposition')

    # plot the colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    add_colorbar(fig, cax, vmin, vmax, unit='', ax=None)

    # plot the trough-locked time
    ax = axs[1]

    lines = ['k--', 'k-', 'k--']
    labels = {
        'std+': r'$\mu+\sigma$',
        'std-': r'$\mu-\sigma$',
        'ste+': r'$\mu+\sigma/\sqrt{n}$',
        'ste-': r'$\mu-\sigma/\sqrt{n}$',
        'mean': r'$\mu$',
    }

    n_signals, n_percentiles, n_points = evoked_time.shape
    t = (np.arange(n_points) - n_points // 2) / float(fs)
    for i, p in enumerate(percentiles):
        label = ('%d %%' % p) if isinstance(p, int) else labels[p]
        ax.plot(t, evoked_time[0, i], lines[i], label=label)

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

    return fig, vmin, vmax, ylim
