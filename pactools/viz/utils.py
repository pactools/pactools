import numpy as np


def compute_ticks(vmin, vmax, unit=''):
    """Compute the ticks in a colorbar."""
    centered = vmin == -vmax

    if vmax == vmin:
        tick_labels = '%s'
        ticks = np.array([vmin])
    else:
        log_scale = np.floor(np.log10((vmax - vmin) / 5.0))
        scale = 10. ** log_scale
        vmax = np.floor(vmax / scale) * scale
        vmin = np.ceil(vmin / scale) * scale

        range_scale = int((vmax - vmin) / scale)
        n_steps = 6
        factors = [6, 8, 4] if centered else [6, 5, 7, 4]
        for i in factors:
            if range_scale % i == 0:
                n_steps = i
                break
        step = np.ceil(range_scale / float(n_steps)) * scale

        tick_labels = '%%.%df' % max(0, -int(log_scale))
        tick_labels += ' ' + unit
        ticks = np.arange(vmin, vmax + step, step)

    return ticks, tick_labels


def add_colorbar(fig, cax, vmin, vmax, unit='', ax=None):
    """Add a colorbar only once in a multiple subplot figure."""
    if vmin == vmax:
        vmin = 0.

    ticks, tick_labels = compute_ticks(vmin, vmax, unit)

    if ax is None:
        fig.subplots_adjust(right=0.81)
        cbar_ax = fig.add_axes([0.86, 0.10, 0.03, 0.8])
    else:
        cbar_ax = None
    cbar = fig.colorbar(cax, ax=ax, cax=cbar_ax, ticks=ticks)
    if unit != '':
        unit = ' ' + unit
    cbar.ax.set_yticklabels([tick_labels % t for t in ticks])



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
