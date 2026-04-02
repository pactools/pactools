"""
Fitting an evoked DAR model
---------------------------
This example creates an artificial signal with phase-amplitude coupling (PAC)
that changes strength over time with a guassian envelope, fits a DAR model
for several signal-to-noise ratios for the evoked PAC and shows the modulation
extracted in the DAR model.

It also shows the comodulogram computed with a DAR model.
"""

import numpy as np
import matplotlib.pyplot as plt

from mne import create_info, Epochs, find_events
from mne.io import RawArray
from pactools import Comodulogram, simulate_pac, raw_to_mask
from pactools.dar_model import DAR


###############################################################################
# Let's define the parameters for the simulation.

fs = 200.  # Hz
high_fq = 50.0  # Hz
low_fq = 5.0  # Hz
low_fq_width = 1.0  # Hz

n_events = 500  # number of epochs (events) simulated
mu = 2.5  # s at peak PAC
sigma = 0.25  # s standard deviation of PAC
trial_len = 4  # s to include for each trial
n_points = int(trial_len * fs)


def gaussian1d(array, mu, sigma):
    return np.exp(-0.5 * ((array - mu) / sigma) ** 2)


###############################################################################
# Let's first create two signals with and without PAC.

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

signal_pac = simulate_pac(n_points=int(50 * fs), fs=fs, high_fq=high_fq,
                          low_fq=low_fq, low_fq_width=low_fq_width,
                          noise_level=0.4, random_state=1)

signal_no_pac = simulate_pac(n_points=int(50 * fs), fs=fs,
                             high_fq=high_fq, low_fq=low_fq,
                             low_fq_width=low_fq_width,
                             noise_level=1., random_state=1)

t = np.linspace(0, 1, int(fs))
axs[0, 0].plot(t, signal_pac[:len(t)])
axs[0, 0].set_title('Signal with PAC (SNR %.1f)' % (1. / 0.4))
axs[0, 1].plot(t, signal_no_pac[:len(t)])
axs[0, 1].set_title('Signal with no PAC (SNR 0.0)')
for ax in axs[0]:
    ax.set_xlabel('time (s)')
    ax.set_ylabel(r'$\mu V$')


for signal, ax in zip([signal_pac, signal_no_pac], axs[1]):
    dar = DAR(ordar=20, ordriv=2, criterion='bic')
    low_fq_range = np.linspace(1, 10, 50)
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, method=dar,
                             progress_bar=False, random_state=0)
    estimator.fit(signal)
    estimator.plot(axs=[ax])


###############################################################################
# Let's look at the Gaussian signal envelope we will be using to look at PAC.

def generate_evoked_pac(noise_level, n_events):
    signal = np.zeros((2, int(n_points * n_events + 10 * fs)))
    events = np.zeros((n_events, 3))
    events[:, 0] = np.arange((5 + mu) * fs,
                             n_points * n_events + (5 + mu) * fs,
                             trial_len * fs, dtype=int)
    events[:, 2] = np.ones((n_events))
    mod_fun = gaussian1d(np.arange(n_points), mu * fs, sigma * fs)
    for i in range(n_events):
        signal_no_pac = simulate_pac(n_points=n_points, fs=fs,
                                     high_fq=high_fq, low_fq=low_fq,
                                     low_fq_width=low_fq_width,
                                     noise_level=1.0, random_state=i)
        signal_pac = simulate_pac(n_points=n_points, fs=fs,
                                  high_fq=high_fq, low_fq=low_fq,
                                  low_fq_width=low_fq_width,
                                  noise_level=noise_level, random_state=i)
        signal[0, i * n_points:(i + 1) * n_points] = \
            signal_pac * mod_fun + signal_no_pac * (1 - mod_fun)
    raw = RawArray(signal, create_info(['Ch1', 'STI 014'], fs,
                                       ['eeg', 'stim']))
    raw.add_events(events, stim_channel='STI 014')
    return raw


raw = generate_evoked_pac(0.4, 100)
events = find_events(raw)
epochs = Epochs(raw, events, tmin=-mu, tmax=trial_len - mu, preload=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=0.4)
ax1.plot(np.linspace(0, trial_len, int(fs * trial_len + 1)),
         gaussian1d(np.arange(n_points + 1), mu * fs, sigma * fs))
ax1.vlines([mu - 3 * sigma, mu + 3 * sigma], ymin=0, ymax=1)
ax1.set_xlabel('time (s)')
ax1.set_ylabel('y-value')
ax1.set_title('1D Gaussian for Convolution with PAC Signal')

ax2.plot(np.linspace(0, trial_len, n_points + 1), epochs._data[0, 0])
ax2.set_xlabel('time (s)')
ax2.set_ylabel(r'$\mu V$')
ax2.set_title('Simulated Evoked Data (One Trial)')

###############################################################################
# Let's now look at how time window and number of events effect evoked PAC
# estimates.

raw = generate_evoked_pac(0.4, n_events)
events = find_events(raw)
n_events_a = np.round(np.logspace(np.log10(10), np.log10(n_events), 5)
                      ).astype(int)

fig, axs = plt.subplots(6, 6, figsize=(20, 20))
fig.subplots_adjust(wspace=0.5)
axs[0, 0].axis('off')
axs[0, 1].set_ylabel('y-value')
for k in range(5):
    axs[1 + k, 0].text(0.5, 0.5, '%i events' % n_events_a[k], fontsize=18)
    axs[1 + k, 0].axis('off')


for j, i in enumerate(range(1, 6)):
    axs[0, j + 1].plot(np.linspace(0, trial_len, int(fs * trial_len)),
                       gaussian1d(np.arange(n_points), mu * fs, sigma * fs))
    axs[0, j + 1].vlines([mu - i * sigma, mu + i * sigma], ymin=0, ymax=1)
    axs[0, j + 1].set_xlabel('time (s)')
    tmin = (mu - i * sigma)
    tmax = (mu + i * sigma)
    tmin_i = int(tmin * fs)
    tmax_i = int(tmax * fs)
    for k in range(5):
        # Comodulogram
        dar = DAR(ordar=20, ordriv=2, criterion='bic')
        low_fq_range = np.linspace(1, 10, 50)
        low_sig, high_sig, mask = raw_to_mask(raw, ixs=[0],
                                              events=events[:n_events_a[k]],
                                              tmin=tmin, tmax=tmax)
        # create the instance of Comodulogram
        estimator = Comodulogram(fs=raw.info['sfreq'],
                                 low_fq_range=np.linspace(1, 10, 20),
                                 low_fq_width=2., method=dar,
                                 progress_bar=True)
        # compute the comodulogram
        estimator.fit(low_sig, high_sig, mask)
        # plot the results
        estimator.plot(axs=[axs[1 + k, j + 1]])


###############################################################################
# Let's now look at how noise level effects evoked PAC estimates.

noise_level_a = np.array([0.25, 0.5, 0.8, 1.0, 1.5])

fig, axs = plt.subplots(6, 6, figsize=(20, 20))
fig.subplots_adjust(wspace=0.5)
axs[0, 0].axis('off')
for k in range(5):
    axs[1 + k, 0].text(0.5, 0.5, '%i events' % n_events_a[k], fontsize=18)
    axs[1 + k, 0].axis('off')
    axs[0, 1 + k].text(0., 0.5, '%.1f SNR' % (1. / noise_level_a[k]),
                       fontsize=18)
    axs[0, 1 + k].axis('off')


tmin = (mu - 3 * sigma)
tmax = (mu + 3 * sigma)
for i in range(5):
    raw = generate_evoked_pac(noise_level_a[i], n_events)
    events = find_events(raw)
    for j in range(5):
        dar = DAR(ordar=20, ordriv=2, criterion='bic')
        low_fq_range = np.linspace(1, 10, 50)
        low_sig, high_sig, mask = raw_to_mask(raw, ixs=[0],
                                              events=events[:n_events_a[j]],
                                              tmin=tmin, tmax=tmax)
        # create the instance of Comodulogram
        estimator = Comodulogram(fs=raw.info['sfreq'],
                                 low_fq_range=np.linspace(1, 10, 20),
                                 low_fq_width=2., method=dar,
                                 progress_bar=True)
        # compute the comodulogram
        estimator.fit(low_sig, high_sig, mask)
        # plot the results
        estimator.plot(axs=[axs[1 + j, 1 + i]])
