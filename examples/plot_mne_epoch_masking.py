"""
Interface with MNE-python
-------------------------
This example shows how to use this package with MNE-python.

It relies on the function ``raw_to_mask``, which takes as input a MNE.Raw
instance and an events array, and returns the corresponding input signals
and masks for the ``Comodulogram.fit`` method.

Note that this example is quite bad since it does not contain any
phase-amplitude coupling. It is just an API example.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt

from pactools import raw_to_mask, Comodulogram, MaskIterator

###############################################################################
# Load the dataset
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

###############################################################################
# Here we define how to build the epochs: which channels will be selected, and
# on which time window around each event.

# select the time interval around the events
tmin, tmax = -5, 15
# select the channels (phase_channel, amplitude_channel)
ixs = (8, 10)

###############################################################################
# Then, we create the inputs with the function raw_to_mask, which creates the
# input arrays and the mask arrays. These arrays are then given to a
# comodulogram instance with the `fit` method, and the `plot` method draws the
# results.

# create the input array for Comodulogram.fit

low_sig, high_sig, mask = raw_to_mask(raw, ixs=ixs, events=events, tmin=tmin,
                                      tmax=tmax)

###############################################################################
# The mask is an iterable which goes over the _unique_ events in the event
# array (if it is 3D). PAC is estimated where the `mask` is `False`.
# Alternatively, we could also compute the `MaskIterator` object directly.
# This is useful if you want to compute PAC on other kinds of time series,
# for example source time courses.

low_sig, high_sig = raw[8, :][0], raw[10, :][0]
mask = MaskIterator(events, tmin, tmax, raw.n_times, raw.info['sfreq'])

# create the instance of Comodulogram
estimator = Comodulogram(fs=raw.info['sfreq'],
                         low_fq_range=np.linspace(1, 10, 20), low_fq_width=2.,
                         method='tort', progress_bar=True)
# compute the comodulogram
estimator.fit(low_sig, high_sig, mask)
# plot the results
estimator.plot(tight_layout=False)
plt.show()
