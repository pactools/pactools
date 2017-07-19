r"""
Phase shift and temporal delay in PAC
-------------------------------------

This example disantangles the two distinct notions of phase shift and temporal
delay in phase-amplitude coupling. The phase shift ($\phi$) is the phase of the
slow oscillation which corresponds to the maximum amplitude of the fast
oscillation. The temporal delay ($\tau$) is the delay between the two coupled
components. The two notions would be identical should the driver be a perfect
stationary sinusoid.

(1st line) When both are equal to zero, the high frequency bursts happen in
the driver's peaks.

(2nd line) When $\tau = 0$ and $\phi != 0$, the bursts are shifted in time with
respect to the driver's peaks, and this shift varies depending on the
instantaneous frequency of the driver.

(3rd line) When $\tau != 0$ and $\phi = 0$, the bursts are shifted in time with
respect to the driver's peaks, and this shift is constant over the signal. In
this case, note how the driver's phase corresponding to the bursts varies
depending on the instantaneous frequency of the driver.

(4th line) Both $\tau$ and $\phi$ can also be both non-zero.

The temporal delay is estimated maximizing the likelihood on DAR models.
The phase shift is extracted from a DAR model fitted with the optimal
temporal delay.

References
==========
Dupre la Tour et al. (2017). Non-linear Auto-Regressive Models for
Cross-Frequency Coupling in Neural Time Series. bioRxiv, 159731.
"""
import numpy as np
import matplotlib.pyplot as plt

from pactools.dar_model import DAR
from pactools.utils import peak_finder
from pactools.utils.viz import phase_string, SEABORN_PALETTES, set_style
from pactools.delay_estimator import DelayEstimator

plt.close('all')
set_style(font_scale=1.4)
blue, green, red, purple, yellow, cyan = SEABORN_PALETTES['deep']

fs = 500.  # Hz
high_fq = 80.0  # Hz
low_fq = 3.0  # Hz
low_fq_mod_fq = 0.5  # Hz
plot_fq_range = [40., 120.]  # Hz

bandwidth = 2.0  # Hz

high_fq_amp = 0.5
low_fq_mod_amp = 3.0

ratio = 1. / 6.
phi_0 = -2 * np.pi * ratio
delay = -1. / low_fq * ratio
offset = -1.
sharpness = 5.
noise_level = 0.1

n_points = 30000
t_plot = 1.  # sec


def sigmoid(array, sharpness):
    return 1. / (1. + np.exp(-sharpness * array))


def clean_peak_finder(sig):
    """Remove first peak if it is at t=0"""
    peak_inds, _ = peak_finder(sig, thresh=None, extrema=1)
    if peak_inds[0] == 0:
        peak_inds = peak_inds[1:]
    return peak_inds


def simulate_and_plot(phi_0, delay, ax, rng):
    """Simulate oscillations with frequency modulation"""
    # create the slow oscillations
    time = np.arange(n_points) / fs
    phase = time * 2 * np.pi * low_fq + np.pi / 2
    # add frequency modulation
    phase += low_fq_mod_amp * np.sin(time * 2 * np.pi * low_fq_mod_fq)
    theta = np.cos(phase)

    # add the fast oscillations
    gamma = np.cos(time * 2 * np.pi * high_fq)
    modulation = sigmoid(offset + np.cos(phase - phi_0),
                         sharpness) * high_fq_amp
    gamma *= modulation

    # add a delay
    delay_point = int(delay * fs)
    gamma = np.roll(gamma, delay_point)
    modulation = np.roll(modulation, delay_point)

    # plot the beginning of the signal
    sel = slice(int(t_plot * fs) + 1)
    lines_theta = ax.plot(time[sel], theta[sel])
    ax.plot(time[sel], gamma[sel])
    lines_modulation = ax.plot(time[sel], modulation[sel])

    # plot the horizontal line of phi_0
    if delay == 0 and False:
        ax.hlines(
            np.cos(-phi_0), time[sel][0], time[sel][-1], color='k',
            linestyle='--')

    gamma_peak_inds = clean_peak_finder(modulation[sel])
    theta_peak_inds = clean_peak_finder(theta[sel])
    cosph_peak_inds = clean_peak_finder(np.cos(phase - phi_0)[sel])

    # plot the vertical lines of the maximum amplitude
    ax.vlines(time[sel][gamma_peak_inds], -1, 1, color='k', linestyle='--')

    # fill vertical intervals between start_idx and stop_idx
    start_idx = gamma_peak_inds
    stop_idx = cosph_peak_inds
    fill_zone = np.zeros_like(time[sel])
    fill_zone[np.minimum(np.maximum(start_idx, 0), sel.stop - 1)] += 1
    fill_zone[np.minimum(np.maximum(stop_idx, 0), sel.stop - 1)] += -1
    ax.fill_between(time[sel], -1, 1, where=np.cumsum(fill_zone) != 0,
                    color=cyan, alpha=0.5)

    # add annotations
    if delay != 0:
        for start, stop in zip(start_idx, stop_idx):
            middle = 0.7 * time[sel][start] + 0.3 * time[sel][stop]
            ax.annotate(r"$\tau_0$", (middle, -1), xycoords='data')
    if phi_0 != 0:
        # ax.annotate(r"$\cos(\phi_0)$", (0, np.cos(phi_0)), xycoords='data')
        ticks = [-1, 0, np.cos(phi_0), 1]
        ticklabels = ['-1', '0', r'$\cos(\phi_0)$', '1']
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)

    # fill the horizontal interval between cos(phi_0) and 1
    ax.fill_between(time[sel], np.cos(phi_0), 1, color=cyan, alpha=0.5)
    # plot the squares of the theta peaks
    ax.plot(time[sel][theta_peak_inds], theta[sel][theta_peak_inds], 's',
            color=lines_theta[0].get_color())
    # plot the circles of maximum gamma amplitude
    ax.plot(time[sel][gamma_peak_inds], modulation[sel][gamma_peak_inds], 'o',
            color=lines_modulation[0].get_color())

    ax.set_xlim([0, t_plot])
    ax.set_xlabel('Time (s)')
    ax.text(0.99, 0.22, r'$\phi_0 = %s$' % (phase_string(phi_0), ),
            horizontalalignment='right', transform=ax.transAxes)
    ax.text(0.99, 0.08, r'$\tau_0 = %.0f \;\mathrm{ms}$' % (delay * 1000, ),
            horizontalalignment='right', transform=ax.transAxes)

    return theta + gamma + noise_level * rng.randn(*gamma.shape)


def fit_dar_and_plot(sig, ax_logl, ax_phase, phi_0, random_state=None):
    dar_model = DAR(ordar=10, ordriv=2)
    est = DelayEstimator(fs, dar_model=dar_model, low_fq=low_fq,
                         low_fq_width=bandwidth,
                         random_state=random_state)
    est.fit(sig)
    est.plot(ax=ax_logl)

    # plot the modulation of the best model
    est.best_model_.plot(ax=ax_phase, mode='c', frange=plot_fq_range)

    ax_phase.set_title('')
    ticks = [-np.pi, phi_0, np.pi]
    ax_phase.set_xticks(ticks)
    ax_phase.set_xticklabels([r'$%s$' % phase_string(d) for d in ticks])
    ax_phase.grid('on')
    ax_phase.grid(color=(0.5, 0.5, 0.5))


# initialize the plots
rng = np.random.RandomState(3)
fig, axs = plt.subplots(4, 3, figsize=(18, 12),
                        gridspec_kw={'width_ratios': [3, 1, 1]})

# loop over the four conditions
for phi_0_, delay_, axs_ in zip([0, phi_0, 0, phi_0], [0, 0, delay, delay],
                                axs):
    sig = simulate_and_plot(phi_0=phi_0_, delay=delay_, ax=axs_[0], rng=rng)
    fit_dar_and_plot(sig, axs_[1], axs_[2], phi_0=phi_0_, random_state=rng)

plt.tight_layout()
plt.show()
