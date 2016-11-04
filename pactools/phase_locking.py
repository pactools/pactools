import numpy as np
import matplotlib.pyplot as plt

from .utils.spectrum import crop_for_fast_hilbert
from .plot_comodulogram import add_colorbar
from .comodulogram import multiple_band_pass


def time_frequency_peak_locking(fs, low_sig, high_sig=None, mask=None,
                                low_fq=6.0,
                                high_fq_range=np.linspace(2.0, 160.0, 80),
                                low_fq_width=2.0,
                                high_fq_width=8.0,
                                t_plot=1.0,
                                method=1,
                                save_name=None,
                                peak_or_trough='peak',
                                draw_peaks=True):
    """
    Plot the theta-trough locked Time-frequency plot of mean power
    modulation time-locked to the theta trough

    Parameters
    ----------
    fs : float,
        Sampling frequency

    low_sig : array, shape (n_epochs, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (n_epochs, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals

    mask : array or None, shape (n_epochs, n_points)
        The locking is only evaluated with the unmasked element of low_sig and
        high_sig. Masking is done after filtering and Hilbert transform.

    low_fq : float
        Filtering frequency (phase signal)

    high_fq_range : array or list
        List of filtering frequencies (amplitude signal)

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    high_fq_width : float
        Bandwidth of the band-pass filter (amplitude signal)

    t_plot : float
        Time to plot around the troughs (in second)

    method : in {0, 1}
        Choose band pass filtering method (in multiple_band_pass)

    save_name : string or None
        Name to use for saving the plot.
        If None, a name is generated based on some parameters.
        If False, the figure is not saved

    peak_or_trough: in {'peak', 'trough'}
        Lock to the maximum (peak) of minimum (trough) of the slow oscillation

    draw_peaks : boolean
        If True, plot the first peaks/troughs in the phase signal

    Return
    ------
    fig : matplotlib.figure.Figure
        Figure instance containing the plot.
    """
    n_cycles = None
    low_fq = np.atleast_1d(low_fq)

    low_sig = np.atleast_2d(low_sig)
    low_sig = crop_for_fast_hilbert(low_sig)
    if mask is not None:
        mask = crop_for_fast_hilbert(mask)

    if high_sig is None:
        high_sig = low_sig

    # compute the slow oscillation
    # n_epochs, n_points = filtered_high.shape
    filtered_low = multiple_band_pass(low_sig, fs,
                                      low_fq, low_fq_width,
                                      method=method)
    filtered_low = filtered_low[0]
    filtered_low_real = np.real(filtered_low)

    if False:
        extrema = 1 if peak_or_trough == 'peak' else -1
        # find the trough in the filtered_low_real with a peak finder
        thresh = (filtered_low_real.max() - filtered_low_real.min()) / 10.
        trough_loc, trough_mag = peak_finder_multi_epochs(filtered_low_real,
                                                          fs=fs, t_plot=t_plot,
                                                          mask=mask,
                                                          thresh=thresh,
                                                          extrema=extrema)
    else:
        # find the trough in the phase
        phase = np.angle(filtered_low)
        if peak_or_trough == 'peak':
            phase = (phase + 2 * np.pi) % (2 * np.pi)
        trough_loc, _ = peak_finder_multi_epochs(phase, fs=fs, t_plot=t_plot,
                                                 mask=mask, extrema=1)
        if draw_peaks:
            trough_mag = filtered_low_real.ravel()[trough_loc]

    # plot the filtered_low_real troughs
    if draw_peaks:
        n_point_plot = min(3000, low_sig.shape[1])
        t = np.arange(n_point_plot) / float(fs)
        plt.figure(figsize=(16, 5))
        plt.plot(t, low_sig[0, :n_point_plot], label='signal')
        plt.plot(t, filtered_low_real[0, :n_point_plot], label='driver')
        plt.plot(trough_loc[trough_loc < n_point_plot] / float(fs),
                 trough_mag[trough_loc < n_point_plot], 'o', label='trough')
        plt.xlabel('Time (sec)')
        plt.title("Driver's trough detection")
        plt.legend(loc=0)

    # extract several signals with band-pass filters
    # n_frequencies, n_epochs, n_points = filtered_high.shape
    filtered_high = multiple_band_pass(high_sig, fs,
                                       high_fq_range, high_fq_width,
                                       n_cycles=n_cycles, method=method)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    axs = axs.ravel()

    # plot the higher part
    plot_trough_locked_time_frequency(
        filtered_high, fs, high_fq_range, trough_loc=trough_loc, t_plot=t_plot,
        mask=mask, fig=fig, ax=axs[0], vmin=None, vmax=None)

    # plot the lower part
    plot_trough_locked_time(
        low_sig, fs, trough_loc=trough_loc,
        t_plot=t_plot, fig=fig, ax=axs[1], ylim=None)

    # save the figure
    if save_name is None:
        save_name = ('%s_wlo%.2f_whi%.1f'
                     % (method, low_fq_width, high_fq_width))
    if save_name:
        fig.savefig(save_name + '.png')

    return fig


def peak_finder_multi_epochs(x0, fs=None, t_plot=None, mask=None,
                             thresh=None, extrema=1, verbose=None):
    """Call peak_finder for multiple epochs, and fill only one array
    as if peak_finder was called with the ravelled array.
    Also remove the peaks that are too close to the start or the end
    of each epoch, and the peaks that are masked by the mask.
    """
    from mne.preprocessing.peak_finder import peak_finder
    n_epochs, n_points = x0.shape

    peak_inds_list = []
    peak_mags_list = []
    for i_epoch in range(n_epochs):
        peak_inds, peak_mags = peak_finder(x0[i_epoch], thresh=thresh,
                                           extrema=extrema, verbose=verbose)

        # remove the peaks too close to the start or the end
        if t_plot is not None and fs is not None:
            n_half_window = int(fs * t_plot / 2.)
            selection = np.logical_and(peak_inds > n_half_window,
                                       peak_inds < n_points - n_half_window)
            peak_inds = peak_inds[selection]
            peak_mags = peak_mags[selection]

        # remove the masked peaks
        if mask is not None:
            selection = mask[i_epoch, peak_inds] == 1
            peak_inds = peak_inds[selection]
            peak_mags = peak_mags[selection]

        peak_inds_list.extend(peak_inds + i_epoch * n_points)
        peak_mags_list.extend(peak_mags)

    if peak_inds_list == []:
        raise ValueError("No %s detected. The signal might be to short, "
                         "or the mask to strong. You can also try to reduce "
                         "the plotted time window `t_plot`."
                         % ["trough", "peak"][(extrema + 1) // 2])

    return np.array(peak_inds_list), np.array(peak_mags_list)


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
                                      trough_loc, t_plot, mask=None,
                                      fig=None, ax=None,
                                      vmin=None, vmax=None):
    """
    Plot the theta-trough locked Time-frequency
    """
    # normalize each signal independently
    n_frequencies, n_epochs, n_points = filtered_high.shape

    # normalization is done everywhere, but mean is computed
    #  only where mask == 1
    if mask is not None:
        masked_filtered_high = filtered_high[:, mask == 1]
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
        masked_filtered_high = filtered_high[:, mask == 1]
    else:
        masked_filtered_high = filtered_high.reshape(n_frequencies, -1)
    mean = masked_filtered_high.mean(axis=1)[:, None, None]
    filtered_high -= mean

    # compute the evoked signals (trough-locked mean)
    evoked_signals = trough_locked_percentile(filtered_high, fs,
                                              trough_loc, t_plot)
    evoked_signals = evoked_signals[:, 0, :]

    if fig is None:
        fig = plt.figure()
        ax = fig.gca()

    # plot
    vmax = np.abs(evoked_signals).max() if vmax is None else vmax
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
    # percentiles = ['mean']
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
