import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def compute_ticks(vmin, vmax, unit=''):
    log_scale = np.floor(np.log10((vmax - vmin) / 3.0))
    scale = 10. ** log_scale
    vmax = np.floor(vmax / scale) * scale
    vmin = np.ceil(vmin / scale) * scale

    range_scale = int((vmax - vmin) / scale)
    step = np.ceil(range_scale / 7.) * scale

    tick_labels = '%%.%df' % max(0, -int(log_scale))
    ticks = np.arange(vmin, vmax + step, step)
    tick_labels += ' ' + unit

    return ticks, tick_labels


def add_colorbar(fig, cax, vmin, vmax, unit='', ax=None):
    """Add a colorbar only once in a multiple subplot figure"""
    if vmin == vmax:
        vmin = 0.

    ticks, tick_labels = compute_ticks(vmin, vmax, unit)

    fig.subplots_adjust(right=0.85)
    if ax is None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.03, 0.8])
    else:
        cbar_ax = None
    cbar = fig.colorbar(cax, ax=ax, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels([tick_labels % t for t in ticks])


def plot_comodulogram_histogram(comodulogram, low_fq_range, low_fq_width,
                                high_fq_range, high_fq_width,
                                method, vmin=None, vmax=None,
                                save_name=None):
    vmin = 0 if vmin is None else vmin
    vmax = comodulogram.max() if vmax is None else vmax
    fig = plt.figure(figsize=(15, 11))
    gs = gridspec.GridSpec(9, 9)
    ax_vert = plt.subplot(gs[:-2, :2])
    ax_hori = plt.subplot(gs[-2:, 2:-1])
    ax_main = plt.subplot(gs[:-2, 2:-1])
    ax_cbar = plt.subplot(gs[:-2, -1])

    extent = [low_fq_range[0], low_fq_range[-1],
              high_fq_range[0], high_fq_range[-1]]
    cax = ax_main.imshow(comodulogram.T, cmap=plt.cm.viridis,
                         aspect='auto', origin='lower', extent=extent,
                         interpolation='none', vmax=vmax, vmin=vmin)

    fig.colorbar(cax, cax=ax_cbar, ticks=np.linspace(vmin, vmax, 6))
    ax_hori.set_xlabel('Phase frequency (Hz) (w=%.1f)' % low_fq_width)
    ax_vert.set_ylabel('Amplitude frequency (Hz) (w=%.2f)' % high_fq_width)
    plt.suptitle('Phase Amplitude Coupling measured with Modulation Index'
                 ' (%s)' % method,
                 fontsize=14)

    ax_vert.plot(np.mean(comodulogram.T, axis=1), high_fq_range)
    ax_vert.set_ylim(extent[2:])
    ax_hori.plot(low_fq_range, np.mean(comodulogram.T, axis=0))
    ax_hori.set_xlim(extent[:2])

    if save_name is None:
        save_name = ('%s_wlo%.2f_whi%.1f'
                     % (method, low_fq_width, high_fq_width))
    fig.savefig(save_name + '.png')
    return fig


def plot_comodulograms(comodulograms, fs, fc_array,
                       titles, fig, axs,
                       cmap=None, vmin=None, vmax=None, unit=''):
    axs = axs.ravel()
    if isinstance(comodulograms, list):
        comodulograms = np.array(comodulograms)

    if comodulograms.ndim == 2:
        comodulograms = comodulograms[None, :, :]

    vmin = 0 if vmin is None else vmin
    vmax = comodulograms.max() if vmax is None else vmax
    cmap = plt.get_cmap('viridis')

    n_channels, n_fc, n_freq = comodulograms.shape
    extend = (fc_array[0], fc_array[-1], 0, fs / 2.)

    # plot the image

    for i in range(n_channels):
        cax = axs[i].imshow(
            comodulograms[i].T, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect='auto', origin='lower', extent=extend, interpolation='none')
        axs[i].set_title(titles[i], fontsize=12)

    axs[-1].set_xlabel('Driver frequency (Hz)')
    axs[0].set_ylabel('Signal frequency (Hz)')
    fig.tight_layout()

    # plot the colorbar once
    ax = axs[0] if len(axs) == 1 else None
    add_colorbar(fig, cax, vmin, vmax, unit=unit, ax=ax)
