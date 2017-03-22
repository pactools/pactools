import numpy as np
import matplotlib.pyplot as plt

from .comodulogram import multiple_band_pass
from .utils.peak_finder import peak_finder
from .viz.plot_phase_locking import plot_trough_locked_time
from .viz.plot_phase_locking import plot_trough_locked_time_frequency


def time_frequency_peak_locking(
        fs, low_sig, high_sig=None, mask=None, low_fq=6.0,
        high_fq_range=np.linspace(10.0, 150.0, 50), low_fq_width=2.0,
        high_fq_width=20.0, t_plot=1.0, filter_method='carrier',
        save_name=None, peak_or_trough='peak', draw_peaks=True, vmin=None,
        vmax=None):
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
        The locking is only evaluated where the mask is False.
        Masking is done after filtering and Hilbert transform.

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

    filter_method : in {'mne', 'carrier'}
        Choose band pass filtering method (in multiple_band_pass)
        'mne': with mne.filter.band_pass_filter
        'carrier': with pactools.Carrier (default)

    save_name : string or None
        Name to use for saving the plot.
        If None, a name is generated based on some parameters.
        If False, the figure is not saved

    peak_or_trough: in {'peak', 'trough'}
        Lock to the maximum (peak) of minimum (trough) of the slow oscillation

    draw_peaks : boolean
        If True, plot the first peaks/troughs in the phase signal

    vmin, vmax:
        Min and max value for the colorbar.

    Return
    ------
    fig : matplotlib.figure.Figure
        Figure instance containing the plot.
    """
    n_cycles = None
    low_fq = np.atleast_1d(low_fq)

    low_sig = np.atleast_2d(low_sig)
    if high_sig is None:
        high_sig = low_sig

    # compute the slow oscillation
    # n_epochs, n_points = filtered_high.shape
    filtered_low = multiple_band_pass(low_sig, fs, low_fq, low_fq_width,
                                      filter_method=filter_method)
    filtered_low = filtered_low[0]
    filtered_low_real = np.real(filtered_low)

    if False:
        extrema = 1 if peak_or_trough == 'peak' else -1
        # find the trough in the filtered_low_real with a peak finder
        thresh = (filtered_low_real.max() - filtered_low_real.min()) / 10.
        trough_loc, trough_mag = peak_finder_multi_epochs(
            filtered_low_real, fs=fs, t_plot=t_plot, mask=mask, thresh=thresh,
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
    filtered_high = multiple_band_pass(high_sig, fs, high_fq_range,
                                       high_fq_width, n_cycles=n_cycles,
                                       filter_method=filter_method)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    axs = axs.ravel()

    # plot the higher part
    plot_trough_locked_time_frequency(
        filtered_high, fs, high_fq_range, trough_loc=trough_loc, t_plot=t_plot,
        mask=mask, fig=fig, ax=axs[0], vmin=vmin, vmax=vmax)

    # plot the lower part
    plot_trough_locked_time(low_sig, fs, trough_loc=trough_loc, t_plot=t_plot,
                            fig=fig, ax=axs[1], ylim=None)

    # save the figure
    if save_name is None:
        save_name = ('%s_wlo%.2f_whi%.1f' %
                     (filter_method, low_fq_width, high_fq_width))
    if save_name:
        fig.savefig(save_name + '.png')

    return fig


def peak_finder_multi_epochs(x0, fs=None, t_plot=None, mask=None, thresh=None,
                             extrema=1, verbose=None):
    """Call peak_finder for multiple epochs, and fill only one array
    as if peak_finder was called with the ravelled array.
    Also remove the peaks that are too close to the start or the end
    of each epoch, and the peaks that are masked by the mask.
    """
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
            selection = mask[i_epoch, peak_inds] == 0
            peak_inds = peak_inds[selection]
            peak_mags = peak_mags[selection]

        peak_inds_list.extend(peak_inds + i_epoch * n_points)
        peak_mags_list.extend(peak_mags)

    if peak_inds_list == []:
        raise ValueError("No %s detected. The signal might be to short, "
                         "or the mask to strong. You can also try to reduce "
                         "the plotted time window `t_plot`." %
                         ["trough", "peak"][(extrema + 1) // 2])

    return np.array(peak_inds_list), np.array(peak_mags_list)
