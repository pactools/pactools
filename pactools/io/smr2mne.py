from __future__ import print_function
import warnings
import sys

import numpy as np
import neo.io
from mne.io.array import RawArray
from mne import create_info


def _smr(filename):
    """helper function to read signal from .smr file (Spike2)"""
    with warnings.catch_warnings():  # avoid warning in Spike2IO
        warnings.simplefilter("ignore")
        reader = neo.io.Spike2IO(filename=filename)
        segment = reader.read_segment(lazy=False, cascade=True,)

    n_channels = len(segment.analogsignals)

    fs = None
    sigs = []
    ch_names = []
    n_samples_max = np.amin([a.size for a in segment.analogsignals])
    for i in np.arange(n_channels):
        channel = segment.analogsignals[i]
        # append the sigs
        sigs.append(np.array(channel[:n_samples_max]))
        # append the name
        ch_names.append(channel.name)
        # get the sampling frequency
        if fs is None:
            fs = float(channel.sampling_rate)
        else:
            assert fs == float(channel.sampling_rate)
    sigs = np.asarray(sigs)

    # load event arrays
    csm = np.array(segment.eventarrays[0].times)
    csp = np.array(segment.eventarrays[1].times)
    cso = np.array(segment.eventarrays[2].times)
    events = {'csp': csp, 'csm': csm, 'cso': cso}

    # check the event's names
    names = ['1k CS-', '7k CS+', ' CS off']
    for i in range(3):
        title = segment.eventarrays[i].annotations['title']
        if sys.version_info > (3, 0, 0):
            title = title.decode('ascii')
        assert title == names[i]

    return sigs, fs, events, ch_names


def smr2sig(filename):
    """
    read signal from .smr file (Spike2) and return a dictionary

    Parameters
    ----------
    filename   : name of file

    Returns
    -------
    data : {'sigs' # list of signals
            'fs'   # sampling frequency
            'csp'  # onset for sound with choc
            'csm'  # onset for sound without choc
            'cso'  # offset for sound (onset + 60sec)
            }
    """
    sigs, fs, events, _ = _smr(filename)
    return sigs, fs, events


def smr2mne(filename):
    """
    read signal from .smr file (Spike2) and return mne objects

    Parameters
    ----------
    filename   : name of file

    Returns
    -------
    raw    : mne.Raw instance
    events : mne compatible events

    Example
    -------
    from mne import Epochs
    raw, events = smr2mne(filename)
    picks = [0, 1, 2, 3, 6, 7]
    epochs = Epochs(raw, events, event_id=+1, tmin=-1, tmax=1, picks=picks)
    epochs.plot(picks=picks, scalings={'misc': 1.0}, n_channels=3, n_epochs=3)
    """
    sigs, fs, events, ch_names = _smr(filename)
    raw, events = sig2mne(sigs, fs, events, ch_names)
    return raw, events


def sig2mne(sigs, fs, events, ch_names=None):
    """
    take a list of signals and events and return mne objects

    Parameters
    ----------
    sigs     : list of signals
    fs       : sampling frequency (float)
    events   :
        csp   : onset for sound with choc
        csm   : onset for sound without choc
        cso   : offset for sound (onset + 60sec)
    ch_names : list of channel names

    Returns
    -------
    mne_raw    : mne.Raw instance
    mne_events : mne compatible events
    """
    if ch_names is None:
        ch_names = len(sigs)

    # create mne.Info instance
    info = create_info(ch_names, fs, ch_types='misc')

    def prepare_event(event, before, after):
        event = np.array(event * fs, dtype=np.int32)
        event = np.c_[event,
                      np.ones_like(event) * before,
                      np.ones_like(event) * after]
        return event

    csm = prepare_event(events['csm'], 0, -1)
    csp = prepare_event(events['csp'], 0, 1)
    cso = prepare_event(events['cso'], 0, 0)

    # merge and sort along the first column
    events = np.r_[csm, csp, cso]
    mne_events = events[events[:, 0].argsort()]

    # create mne.Raw instance
    mne_raw = RawArray(data=sigs, info=info)
    return mne_raw, mne_events
