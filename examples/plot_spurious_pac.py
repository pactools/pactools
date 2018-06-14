r"""
Spurious PAC with spikes
------------------------

This example demonstrates how a spike train can generate a significant amount
of phase-amplitude coupling (PAC), even though there is no nested oscillations.
This phenomenon is usually referred as spurious PAC.

References
==========
Dupre la Tour et al. (2017). Non-linear Auto-Regressive Models for
Cross-Frequency Coupling in Neural Time Series. bioRxiv, 159731.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from pactools import Comodulogram, REFERENCES
from pactools.utils.pink_noise import pink_noise
from pactools.utils.validation import check_random_state


def simulate_spurious_pac(n_points, fs, spike_amp=1.5, spike_fwhm=0.01,
                          spike_fq=10., spike_interval_jitter=0.2,
                          random_state=None):
    """Simulate some spurious phase-amplitude coupling (PAC) with spikes

    References
    ----------
    Gerber, E. M., Sadeh, B., Ward, A., Knight, R. T., & Deouell, L. Y. (2016).
    Non-sinusoidal activity can produce cross-frequency coupling in cortical
    signals in the absence of functional interaction between neural sources.
    PloS one, 11(12), e0167351
    """
    n_points = int(n_points)
    fs = float(fs)
    rng = check_random_state(random_state)

    # draw the position of the spikes
    interval_min = 1. / float(spike_fq) * (1. - spike_interval_jitter)
    interval_max = 1. / float(spike_fq) * (1. + spike_interval_jitter)
    n_spikes_max = np.int(n_points / fs / interval_min)
    spike_intervals = rng.uniform(low=interval_min, high=interval_max,
                                  size=n_spikes_max)
    spike_positions = np.cumsum(np.int_(spike_intervals * fs))
    spike_positions = spike_positions[spike_positions < n_points]

    # build the spike time series, using a convolution
    spikes = np.zeros(n_points)
    spikes[spike_positions] = spike_amp
    # full width at half maximum to standard deviation convertion
    spike_std = spike_fwhm / (2 * np.sqrt(2 * np.log(2)))
    spike_shape = scipy.signal.gaussian(M=np.int(spike_std * fs * 10),
                                        std=spike_std * fs)
    spikes = scipy.signal.fftconvolve(spikes, spike_shape, mode='same')

    noise = pink_noise(n_points, slope=1., random_state=random_state)

    return spikes + noise, spikes


#############################################################################
# Let's generate the spurious PAC signal
fs = 1000.  # Hz
n_points = 60000
spike_amp = 1.5
random_state = 0

# generate the signal
signal, spikes = simulate_spurious_pac(
    n_points=n_points, fs=fs, random_state=random_state, spike_amp=spike_amp)

# plot the signal and the spikes
n_points_plot = np.int(1. * fs)
time = np.arange(n_points_plot) / fs
fig, axs = plt.subplots(nrows=2, figsize=(10, 6), sharex=True)
axs[0].plot(time, signal[:n_points_plot], color='C0')
axs[0].set(title='spikes + pink noise')
axs[1].plot(time, spikes[:n_points_plot], color='C1')
axs[1].set(xlabel='Time (sec)', title='spikes')
plt.show()

#############################################################################
# Compute the comodulograms

methods = ['ozkurt', 'penny', 'tort', 'duprelatour']
low_fq_range = np.linspace(1., 20., 60)
high_fq_range = np.linspace(low_fq_range[-1], 100., 60)
low_fq_width = 4.  # Hz

# A good rule of thumb is n_surrogates = 10 / p_value. Example: 10 / 0.05 = 200
# Here we use 10 to be fast
n_surrogates = 10
p_value = 0.05
n_jobs = 11

# prepare the plot axes
n_lines = 2
n_columns = int(np.ceil(len(methods) / float(n_lines)))
figsize = (4 * n_columns, 3 * n_lines)
fig, axs = plt.subplots(nrows=n_lines, ncols=n_columns, figsize=figsize)
axs = axs.ravel()

for ax, method in zip(axs, methods):
    # compute the comodulograms
    estimator = Comodulogram(fs=fs, low_fq_range=low_fq_range,
                             high_fq_range=high_fq_range,
                             low_fq_width=low_fq_width, method=method,
                             n_surrogates=n_surrogates, n_jobs=n_jobs)
    estimator.fit(signal)

    # plot the comodulogram with contour levels
    estimator.plot(contour_method='comod_max', contour_level=p_value, axs=[ax],
                   titles=[REFERENCES[method]])

fig.tight_layout()
plt.show()
