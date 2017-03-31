import mne
import numpy as np
import matplotlib.pyplot as plt
from mne import io
from mne.datasets import sample

from pactools import raw_to_mask, Comodulogram

event_id = {'Visual/Left': 3}
tmin, tmax = -0.2, 0.5

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')

raw = io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

low_sig, high_sig, mask = raw_to_mask(raw, ixs=(0, 1), events=events,
                                      tmin=-10., tmax=+30.)
estimator = Comodulogram(fs=raw.info['sfreq'],
                         low_fq_range=np.linspace(1, 10, 20), low_fq_width=2.,
                         method='tort', progress_bar=True)
estimator.fit(low_sig, high_sig, mask)
estimator.plot()
plt.show()
