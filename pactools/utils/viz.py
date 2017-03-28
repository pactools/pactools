import itertools
import matplotlib as mpl
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


SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#55A868", "#C44E52",
          "#8172B2", "#CCB974", "#64B5CD"],
    muted=["#4878CF", "#6ACC65", "#D65F5F",
           "#B47CC7", "#C4AD66", "#77BEDB"],
    pastel=["#92C6FF", "#97F0AA", "#FF9F9A",
            "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    bright=["#003FFF", "#03ED3A", "#E8000B",
            "#8A2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#017517", "#8C0900",
          "#7600A1", "#B8860B", "#006374"],
    colorblind=["#0072B2", "#009E73", "#D55E00",
                "#CC79A7", "#F0E442", "#56B4E9"]
    )


# inspired by seaborn
def mpl_palette(name, n_colors=6, extrema=False, cycle=False):
    """Return discrete colors from a matplotlib palette.
    Note that this handles the qualitative colorbrewer palettes
    properly, although if you ask for more colors than a particular
    qualitative palette can provide you will get fewer than you are
    expecting.

    Parameters
    ----------
    name : string
        Name of the palette. This should be a named matplotlib colormap.
    n_colors : int
        Number of discrete colors in the palette.
    extrema : boolean
        If True, include the extrema of the palette.
    cycle : boolean
        If True, return a itertools.cycle.

    Returns
    -------
    palette : colormap or itertools.cycle
        List-like object of colors as RGB tuples
    """
    if name in SEABORN_PALETTES:
        palette = SEABORN_PALETTES[name]
        # Always return as many colors as we asked for
        pal_cycle = itertools.cycle(palette)
        palette = [next(pal_cycle) for _ in range(n_colors)]

    elif name in dir(mpl.cm) or name[:-2] in dir(mpl.cm):
        cmap = getattr(mpl.cm, name)

        if extrema:
            bins = np.linspace(0, 1, n_colors)
        else:
            bins = np.linspace(0, 1, n_colors * 2 - 1 + 2)[1:-1:2]
        palette = list(map(tuple, cmap(bins)[:, :3]))

    else:
        raise ValueError("%s is not a valid palette name" % name)

    if cycle:
        return itertools.cycle(palette)
    else:
        return palette


def phase_string(sig):
    """Take the angle and create a string as \pi if close to some \pi values"""
    if isinstance(sig, np.ndarray):
        sig = sig.ravel()[0]
    if isinstance(sig, np.complex):
        angle = np.angle(sig)
    else:
        angle = sig

    pi_multiples = np.pi * np.arange(-1, 1.25, 0.25)
    strings = [
        '-\pi', '-3\pi/4', '-\pi/2', '-\pi/4', '0', '\pi/4', '\pi/2', '3\pi/4',
        '\pi'
    ]

    if np.min(np.abs(angle - pi_multiples)) < 1e-3:
        return strings[np.argmin(np.abs(angle - pi_multiples))]
    else:
        return '%.2f' % angle
