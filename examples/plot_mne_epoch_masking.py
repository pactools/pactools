"""
Interface with MNE-python
-------------------------
This example shows how to use this package with MNE-python.

It relies on the function ``raw_to_mask``, which takes as input a MNE.Raw
instance and an events array, and returns the corresponding input signals
and masks for the ``Comodulogram.fit`` method.

Note that this example is quite bad since it does not contain any
phase-amplitude coupling.
"""
import mne
import numpy as np
import matplotlib.pyplot as plt

from pactools import raw_to_mask, Comodulogram

# load the dataset
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

# select the time interval around the events
tmin, tmax = -5, 15
# select the channels (phase_channel, amplitude_channel)
ixs = (8, 10)

# create the input array for Comodulogram.fit
low_sig, high_sig, mask = raw_to_mask(raw, ixs=ixs, events=events, tmin=tmin,
                                      tmax=tmax)
# create the instance of Comodulogram
estimator = Comodulogram(fs=raw.info['sfreq'],
                         low_fq_range=np.linspace(1, 10, 20), low_fq_width=2.,
                         method='tort', progress_bar=True)
# compute the comodulogram
estimator.fit(low_sig, high_sig, mask)
# plot the results
estimator.plot(tight_layout=False)
plt.show()
