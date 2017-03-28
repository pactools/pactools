import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from .comodulogram import multiple_band_pass
from .utils.peak_finder import peak_finder
from .utils.validation import check_consistent_shape, check_array
from .utils.viz import add_colorbar, mpl_palette


class PeakLocking(object):
    """An object to compute time average and time-frequency averaged with
    peak-locking, to analyze phase-amplitude coupling.

    Parameters
    ----------
    fs : float
        Sampling frequency

    low_fq : float
        Filtering frequency (phase signal)

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    high_fq_range : array or list, shape (n_high, ), or 'auto'
        List of filtering frequencies (amplitude signal)
        If 'auto', it uses np.linspace(low_fq, fs / 2, 40).

    high_fq_width : float or 'auto'
        Bandwidth of the band-pass filter (amplitude signal)
        If 'auto', it uses 2 * low_fq.

    t_plot : float
        Time to plot around the peaks (in second)

    filter_method : in {'mne', 'carrier'}
        Choose band pass filtering method (in multiple_band_pass)
        'mne': with mne.filter.band_pass_filter
        'carrier': with pactools.Carrier (default)

    peak_or_trough: in {'peak', 'trough'}
        Lock to the maximum (peak) of minimum (trough) of the slow
        oscillation.

    percentiles : list of float or string, shape (n_percentiles, )
        Percentile to compute for the time representation.
        It can also include 'mean', 'std' or 'ste'
        (resp. mean, standard deviation or standard error).
    """
    def __init__(self, fs, low_fq, low_fq_width=1.0, high_fq_range='auto',
                 high_fq_width='auto', t_plot=1.0, filter_method='carrier',
                 peak_or_trough='peak', percentiles=['std+', 'mean', 'std-']):
        self.fs = fs
        self.low_fq = low_fq
        self.high_fq_range = high_fq_range
        self.low_fq_width = low_fq_width
        self.high_fq_width = high_fq_width
        self.t_plot = t_plot
        self.filter_method = filter_method
        self.peak_or_trough = peak_or_trough
        self.percentiles = percentiles

    def fit(self, low_sig, high_sig=None, mask=None):
        """
        Compute peak-locked time-averaged and time-frequency representations.

        Parameters
        ----------
        low_sig : array, shape (n_epochs, n_points)
            Input data for the phase signal

        high_sig : array or None, shape (n_epochs, n_points)
            Input data for the amplitude signal.
            If None, we use low_sig for both signals

        mask : array or None, shape (n_epochs, n_points)
            The locking is only evaluated where the mask is False.
            Masking is done after filtering and Hilbert transform.

        ax_draw_peaks : boolean or matplotlib.axes.Axes instance
            If True, plot the first peaks/troughs in the phase signal
            Plot on the matplotlib.axes.Axes instance if given, or on a
            new figure.

        Attributes
        ----------
        time_frequency_ : array, shape (n_high, n_window)
            Time-frequency representation, averaged with peak-locking.
            (n_window is the number of point in t_plot seconds)

        time_average_ : array, shape (n_percentiles, n_window)
            Time representation, averaged with peak-locking.
            (n_window is the number of point in t_plot seconds)
        """
        self.low_fq = np.atleast_1d(self.low_fq)
        if self.high_fq_range == 'auto':
            self.high_fq_range = np.linspace(self.low_fq[0], self.fs / 2.0, 40)
        if self.high_fq_width == 'auto':
            self.high_fq_width = 2 * self.low_fq[0]
        self.high_fq_range = np.asarray(self.high_fq_range)

        self.low_sig = check_array(low_sig)
        self.high_sig = check_array(high_sig, accept_none=True)
        if self.high_sig is None:
            self.high_sig = self.low_sig

        self.mask = check_array(mask, accept_none=True)
        check_consistent_shape(self.low_sig, self.high_sig, self.mask)

        # compute the slow oscillation
        # 1, n_epochs, n_points = filtered_low.shape
        filtered_low = multiple_band_pass(low_sig, self.fs, self.low_fq,
                                          self.low_fq_width,
                                          filter_method=self.filter_method)
        self.filtered_low_ = filtered_low[0]
        filtered_low_real = np.real(self.filtered_low_)

        if False:
            # find the peak in the filtered_low_real
            extrema = 1 if self.peak_or_trough == 'peak' else -1
            thresh = (filtered_low_real.max() - filtered_low_real.min()) / 10.
            self.peak_loc, self.peak_mag = peak_finder_multi_epochs(
                filtered_low_real, fs=self.fs, t_plot=self.t_plot,
                mask=self.mask, thresh=thresh, extrema=extrema)
        else:
            # find the peak in the phase of filtered_low
            phase = np.angle(self.filtered_low_)
            if self.peak_or_trough == 'peak':
                phase = (phase + 2 * np.pi) % (2 * np.pi)
            self.peak_loc, _ = peak_finder_multi_epochs(
                phase, fs=self.fs, t_plot=self.t_plot, mask=self.mask,
                extrema=1)
            self.peak_mag = filtered_low_real.ravel()[self.peak_loc]

        # extract several signals with band-pass filters
        # n_high, n_epochs, n_points = filtered_high.shape
        self.filtered_high_ = multiple_band_pass(
            self.high_sig, self.fs, self.high_fq_range, self.high_fq_width,
            n_cycles=None, filter_method=self.filter_method)

        # compute the peak locked time-frequency representation
        time_frequency_ = peak_locked_time_frequency(
            self.filtered_high_, self.fs, self.high_fq_range,
            peak_loc=self.peak_loc, t_plot=self.t_plot, mask=self.mask)

        # compute the peak locked time representation
        # we don't need the mask here, since only the valid peak locations are
        # kept in peak_finder_multi_epochs
        time_average_ = peak_locked_percentile(self.low_sig[None, :], self.fs,
                                               self.peak_loc, self.t_plot,
                                               self.percentiles)
        time_average_ = time_average_[0, :, :]

        self.time_frequency_ = time_frequency_
        self.time_average_ = time_average_

        return self

    def plot_peaks(self, ax=None):
        # plot the filtered_low_real peaks
        if not isinstance(ax, matplotlib.axes.Axes):
            fig = plt.figure(figsize=(16, 5))
            ax = fig.gca()
        n_point_plot = min(3000, self.low_sig.shape[1])
        time = np.arange(n_point_plot) / float(self.fs)

        filtered = np.real(self.filtered_low_[0, :n_point_plot])

        ax.plot(time, self.low_sig[0, :n_point_plot], label='signal')
        ax.plot(time, filtered, label='driver')
        ax.plot(self.peak_loc[self.peak_loc < n_point_plot] / float(self.fs),
                self.peak_mag[self.peak_loc < n_point_plot], 'o',
                label='peaks')
        ax.set_xlabel('Time (sec)')
        ax.set_title("Driver's peak detection")
        ax.set_legend(loc=0)

    def plot(self, axs=None, vmin=None, vmax=None, ylim=None):
        """
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

        # plot the peak-locked time-frequency
        ax = axs[0]
        n_high, n_points = self.time_average_.shape
        vmax = np.abs(self.time_frequency_).max() if vmax is None else vmax
        vmin = -vmax
        extent = (
            -self.t_plot / 2,
            self.t_plot / 2,
            self.high_fq_range[0],
            self.high_fq_range[-1], )
        cax = ax.imshow(self.time_frequency_, cmap=plt.get_cmap('RdBu_r'),
                        vmin=vmin, vmax=vmax, aspect='auto', origin='lower',
                        interpolation='none', extent=extent)

        # ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Driver peak-locked Time-frequency decomposition')

        # plot the colorbar
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        add_colorbar(fig, cax, vmin, vmax, unit='', ax=None)

        # plot the peak-locked time
        ax = axs[1]

        labels = {
            'std+': r'$\mu+\sigma$',
            'std-': r'$\mu-\sigma$',
            'ste+': r'$\mu+\sigma/\sqrt{n}$',
            'ste-': r'$\mu-\sigma/\sqrt{n}$',
            'mean': r'$\mu$',
        }
        colors = mpl_palette('viridis', n_colors=len(self.percentiles))

        n_percentiles, n_points = self.time_average_.shape
        time = (np.arange(n_points) - n_points // 2) / float(self.fs)
        for i, p in enumerate(self.percentiles):
            label = ('%d %%' % p) if isinstance(p, int) else labels[p]
            ax.plot(time, self.time_average_[i, :], color=colors[i],
                    label=label)

        ax.set_xlabel('Time (sec)')
        ax.set_title('Driver peak-locked average of raw signal')
        ax.legend(loc='lower center', ncol=5, labelspacing=0.)
        ax.grid('on')

        # make room for the legend or apply specified ylim
        if ylim is None:
            ylim = ax.get_ylim()
            ylim = (ylim[0] - (ylim[1] - ylim[0]) * 0.2, ylim[1])
        ax.set_ylim(ylim)

        return fig


def peak_finder_multi_epochs(x0, fs=None, t_plot=None, mask=None, thresh=None,
                             extrema=1):
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
                                           extrema=extrema)

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


def peak_locked_time_frequency(filtered_high, fs, high_fq_range, peak_loc,
                               t_plot, mask=None):
    """
    Compute the peak-locked Time-frequency
    """
    # normalize each signal independently
    n_high, n_epochs, n_points = filtered_high.shape

    # normalization is done everywhere, but mean is computed
    #  only where mask == 1
    if mask is not None:
        masked_filtered_high = filtered_high[:, mask == 0]
    else:
        masked_filtered_high = filtered_high.reshape(n_high, -1)

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
        masked_filtered_high = filtered_high.reshape(n_high, -1)
    mean = masked_filtered_high.mean(axis=1)[:, None, None]
    filtered_high -= mean

    # compute the evoked signals (peak-locked mean)
    evoked_signals = peak_locked_percentile(filtered_high, fs, peak_loc,
                                            t_plot)
    evoked_signals = evoked_signals[:, 0, :]

    return evoked_signals


def peak_locked_percentile(signals, fs, peak_loc, t_plot,
                           percentiles=['mean']):
    """
    Compute the mean of each signal in signals, locked to the peaks

    Parameters
    ----------
    signals    : shape (n_signals, n_epochs, n_points)
    fs         : sampling frequency
    peak_loc   : indices of the peak locations
    t_plot     : in second, time to plot around the peaks
    percentiles: list of precentile to compute. It can also include 'mean',
                 'std' or 'ste' (mean, standard deviation or standard error).

    Returns
    -------
    evoked_signals : array, shape (n_signals, n_percentiles, n_window)
    """
    n_signals, n_epochs, n_points = signals.shape
    n_window = int(fs * t_plot / 2.) * 2 + 1
    n_percentiles = len(percentiles)

    # build indices matrix: each line is a index range around peak locations
    indices = np.tile(peak_loc, (n_window, 1)).T
    indices = indices + np.arange(n_window) - n_window // 2

    # ravel the epochs since we now have isolated events
    signals = signals.reshape(n_signals, -1)

    n_peaks = indices.shape[0]
    assert n_peaks > 0

    # compute the evoked signals (peak-locked mean)
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
                    evoked_signals[i_s, i_p] = mean + std / np.sqrt(n_peaks)
                elif p == 'ste-':
                    evoked_signals[i_s, i_p] = mean - std / np.sqrt(n_peaks)
                else:
                    raise (ValueError, 'wrong percentile string: %s' % p)

    return evoked_signals
