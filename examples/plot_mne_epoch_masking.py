"""
Interface with MNE-python
-------------------------
This example shows how to use this package with MNE-python.

It relies on the function ``raw_to_mask``, which takes as input a MNE.Raw
instance and an events array, and returns the corresponding input signals
and masks for the ``Comodulogram.fit`` method.
"""
import mne
import numpy as np
import os.path as op
import matplotlib.pyplot as plt

from pactools import simulate_pac, raw_to_mask, Comodulogram, MaskIterator

###############################################################################
# Simulate a dataset and save it out

n_events = 100
mu = 1.  # mean onset of PAC in seconds
sigma = 0.25  # standard deviation of onset of PAC in seconds
trial_len = 2.  # len of the simulated trial in seconds
first_samp = 5  # seconds before the first sample and after the last

fs = 200.  # Hz
high_fq = 50.0  # Hz
low_fq = 3.0  # Hz
low_fq_width = 2.0  # Hz

n_points = int(trial_len * fs)
noise_level = 0.4

np.random.seed(0)


def gaussian1d(array, mu, sigma):
    return np.exp(-0.5 * ((array - mu) / sigma) ** 2)


# leave one channel for events and make signal as long as events
# with a bit of room on either side so events don't get cut off
signal = np.zeros((7, int(n_points * n_events + 2 * first_samp * fs)))
events = np.zeros((n_events, 3), dtype=int)
events[:, 0] = np.arange((first_samp + mu) * fs,
                         n_points * n_events + (first_samp + mu) * fs,
                         trial_len * fs, dtype=int)
events[:, 2] = np.ones((n_events))
mod_fun = gaussian1d(np.arange(n_points), mu * fs, sigma * fs)
for i in range(n_events):
    signal_no_pac = np.random.randn(n_points)
    driver, carrier = simulate_pac(n_points=n_points, fs=fs,
                                   high_fq=high_fq, low_fq=low_fq,
                                   low_fq_width=low_fq_width,
                                   noise_level=noise_level,
                                   separate=True, random_state=i)
    signal[0, i * n_points:(i + 1) * n_points] = \
        driver * mod_fun + signal_no_pac * (1 - mod_fun)
    signal[1, i * n_points:(i + 1) * n_points] = \
        carrier * mod_fun + signal_no_pac * (1 - mod_fun)
    # note: extra channels are necessary to properly separate driver
    # and carrier signals due to average reference
    for j in range(2, 6):
        signal[j, i * n_points:(i + 1) * n_points] = np.random.randn(n_points)


ch_names = ['Driver', 'Carrier']
ch_names += ['Ch%i' % i for i in range(2, 6)]
ch_names += ['STI 014']
info = mne.create_info(ch_names, fs, ['eeg'] * 6 + ['stim'])
raw = mne.io.RawArray(signal, info)
raw.add_events(events, stim_channel='STI 014')

###############################################################################
# Let's plot the signal and its power spectral density to visualize the data.
# As shown in the plots, there is a peak at the driver frequency at
# 3 Hz in the 'Driver' channel and a peak at the carrier frequency at 50 Hz
# in the 'Carrier' channel.

raw.plot_psd(fmax=60)

###############################################################################
# In the evoked plot of the 'Driver' and 'Carrier' channels, you can see what
# phase-amplitude coupling looks like in the voltage trace.

epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=2)
epochs.average().plot(picks=['Driver', 'Carrier'])

###############################################################################
# Let's save the raw object out for input/output demonstration purposes

root = mne.utils._TempDir()
raw.save(op.join(root, 'pac_example-raw.fif'))

###############################################################################
# Here we define how to build the epochs: which channels will be selected, and
# on which time window around each event.

raw = mne.io.Raw(op.join(root, 'pac_example-raw.fif'))
events = mne.find_events(raw)

# select the time interval around the events
tmin, tmax = mu - 3 * sigma, mu + 3 * sigma
# select the channels (phase_channel, amplitude_channel)
ixs = (0, 1)

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

low_sig, high_sig = raw[ixs[0], :][0], raw[ixs[1], :][0]
mask = MaskIterator(events, tmin, tmax, raw.n_times, raw.info['sfreq'])

# create the instance of Comodulogram
estimator = Comodulogram(fs=raw.info['sfreq'],
                         low_fq_range=np.linspace(1, 10, 20), low_fq_width=2.,
                         method='duprelatour', progress_bar=True)
# compute the comodulogram
estimator.fit(low_sig, high_sig, mask)
# plot the results
estimator.plot(tight_layout=False)
plt.show()
