import numpy as np
from mne.io import BaseRaw


def raw_to_mask(raw, ixs, events=None, tmin=None, tmax=None):
    """
    Warning: events is stored in indices, tmin and tmax are stored in seconds.

    Parameters
    ----------
    raw : an instance of Raw, containing data of shape (n_channels, n_times)
        The data used to calculate PAC

    ixs : int or couple of int
        The indices for the low/high frequency channels. If only one is given,
        the same channel is used for both low_sig and high_sig.

    events : array, shape (n_events, 3) | array, shape (n_events,) | None
        MNE events array. To be supplied if data is 2D and output should be
        split by events. In this case, `tmin` and `tmax` must be provided. If
        `ndim == 1`, it is assumed to be event indices, and all events will be
        grouped together. Otherwise, events will be grouped along the third
        dimension.

    tmin : float | list of floats, shape (n_windows, ) | None
        If `events` is not provided, it is the start time to use in `raw`.
        If `events` is provided, it is the time (in seconds) to include before
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `min(raw.times)`.

    tmax : float | list of floats, shape (n_windows, ) | None
        If `events` is not provided, it is the stop time to use in `raw`.
        If `events` is provided, it is the time (in seconds) to include after
        each event index. If a list of floats is given, then PAC is calculated
        for each pair of `tmin` and `tmax`. Defaults to `max(raw.times)`.

    Attributes
    ----------
    low_sig : array, shape (1, n_points)
        Input data for the phase signal

    high_sig : array or None, shape (1, n_points)
        Input data for the amplitude signal.
        If None, we use low_sig for both signals.

    mask : MaskIterator instance
        Object that behaves like a list of mask, without storing them all.
        The PAC will only be evaluated where the mask is False.

    Examples
    --------
    >>> low_sig, high_sig, mask = raw_to_mask(raw, ixs, events, tmin, tmax)
    >>> n_masks = len(mask)
    >>> for one_mask in mask:
    >>>     pass
    """
    if not isinstance(raw, BaseRaw):
        raise ValueError('Must supply Raw as input')

    ixs = np.atleast_1d(ixs)
    fs = raw.info['sfreq']

    data = raw[:][0]
    n_channels, n_points = data.shape
    low_sig = data[ixs[0]][None, :]
    if ixs.shape[0] > 1:
        high_sig = data[ixs[1]][None, :]
    else:
        high_sig = None

    mask = MaskIterator(events, tmin, tmax, n_points, fs)

    return low_sig, high_sig, mask


class MaskIterator(object):
    """Iterator that creates the masks one at a time.

    Examples
    --------
    >>> all_masks = MaskIterator(events, tmin, tmax, n_points, fs)
    >>> n_masks = len(all_masks)
    >>> for one_mask in all_masks:
    >>>     pass
    """

    def __init__(self, events, tmin, tmax, n_points, fs):
        self.events = events
        self.tmin = tmin
        self.tmax = tmax
        self.n_points = n_points
        self.fs = float(fs)
        self._init()

    def _init(self):
        self.tmin = np.atleast_1d(self.tmin)
        self.tmax = np.atleast_1d(self.tmax)

        if len(self.tmin) != len(self.tmax):
            raise ValueError('tmin and tmax have differing lengths')
        n_windows = len(self.tmin)

        if self.events is None:
            self.events = np.array([0.])
            n_events = 1

        if self.events.ndim == 1:
            n_events = 1  # number of different event kinds
        else:
            n_events = np.unique(self.events[:, -1]).shape[0]

        self._n_iter = n_windows * n_events

    def __iter__(self):
        return self.next()

    def __len__(self):
        return self._n_iter

    def next(self):
        if self.events.ndim == 1:
            event_names = [None, ]
        else:
            event_names = np.unique(self.events[:, -1])

        mask = np.empty((1, self.n_points), dtype=bool)
        for event_name in event_names:

            if self.events.ndim == 1:
                # select all the events since their kind is not specified
                these_events = self.events
            else:
                # select the event indices of one kind of event
                these_events = self.events[self.events[:, -1] == event_name, 0]

            for tmin, tmax in zip(self.tmin, self.tmax):
                mask.fill(True)  # it masks everything
                for event in these_events:
                    start, stop = None, None
                    if tmin is not None:
                        start = int(event + tmin * self.fs)
                    if tmax is not None:
                        stop = int(event + tmax * self.fs)
                    mask[:, start:stop] = False

                yield mask
