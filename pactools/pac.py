import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.random import RandomState
from scipy.signal import hilbert
from scipy import linalg
from mne.filter import band_pass_filter

from .utils.progress_bar import ProgressBar
from .utils.spectrum import crop_for_fast_hilbert
from .utils.carrier import Carrier


def multiple_band_pass(sig, fs, f_range, f_width, n_cycles=None, method=1):
    """
    Band-pass filter the signal at multiple frequencies
    """
    fixed_n_cycles = n_cycles
    sig = crop_for_fast_hilbert(sig)
    f_range = np.atleast_1d(f_range)

    if method == 1:
        fir = Carrier()

    n_frequencies = f_range.shape[0]
    res = np.zeros((n_frequencies, sig.shape[0]), dtype=np.complex128)
    for i, f in enumerate(f_range):
        # evaluate the number of cycle for this f_width and f
        if fixed_n_cycles is None:
            n_cycles = 1.65 * f / f_width

        # 0--------- with mne.filter.band_pass_filter
        if method == 0:
            low_sig = band_pass_filter(
                sig, Fs=fs, Fp1=f - f_width / 2.0, Fp2=f + f_width / 2.0,
                l_trans_bandwidth=f_width / 4.0,
                h_trans_bandwidth=f_width / 4.0,
                n_jobs=1, method='iir')

        # 1--------- with pactools.Carrier
        if method == 1:
            fir.design(fs, f, n_cycles, None, zero_mean=True)
            low_sig = fir.direct(sig)

        # common to the two methods
        res[i, :] = hilbert(low_sig)

    return res


def norm(x):
    """Compute the Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
    """
    x = np.asarray(x)

    if np.any(np.iscomplex(x)):
        return np.sqrt(squared_norm(x))
    else:
        nrm2, = linalg.get_blas_funcs(['nrm2'], [x])
        return nrm2(x)


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = x.ravel(order='K')
    return np.dot(x, np.conj(x))


def _modulation_index(filtered_low, filtered_high, method, fs, n_surrogates,
                      progress_bar, draw_phase):
    """
    Helper for modulation_index
    """
    rng = RandomState(42)

    tmax = filtered_low.shape[1]
    n_low = filtered_low.shape[0]
    n_high = filtered_high.shape[0]

    # phase of the low frequency signals
    for i in range(n_low):
        filtered_low[i] = np.angle(filtered_low[i])
    filtered_low = np.real(filtered_low)

    # amplitude of the high frequency signals
    norm_a = np.zeros(n_high)
    for j in range(n_high):
        filtered_high[j] = np.abs(filtered_high[j])
        if method == 'ozkurt':
            norm_a[j] = norm(filtered_high[j])
    filtered_high = np.real(filtered_high)

    # Calculate the modulation index for each couple
    if progress_bar:
        bar = ProgressBar('modulation index', max_value=n_low)
    MI = np.zeros((n_low, n_high))
    exp_phase = None
    for i in range(n_low):
        if method != 'tort':
            exp_phase = np.exp(1j * filtered_low[i])

        for j in range(n_high):
            MI[i, j] = _one_modulation_index(
                amplitude=filtered_high[j], phase=filtered_low[i],
                exp_phase=exp_phase, norm_a=norm_a[j], method=method,
                tmax=tmax, fs=fs, n_surrogates=n_surrogates,
                random_state=rng, draw_phase=draw_phase)

        if progress_bar:
            bar.update(i + 1)

    return MI


def _one_modulation_index(amplitude, phase, exp_phase, norm_a, method,
                          tmax, fs, n_surrogates, random_state, draw_phase):
    if method == 'ozkurt':
        # Modulation index as in Ozkurt 2011
        MI = np.abs(np.mean(amplitude * exp_phase))

        MI /= norm_a
        MI *= np.sqrt(tmax)

    elif method == 'canolty':
        # Modulation index as in Canolty 2006
        MI = np.abs(np.mean(amplitude * exp_phase))

        if n_surrogates > 0:
            # compute surrogates MIs
            MI_surr = np.empty(n_surrogates)
            for s in range(n_surrogates):
                shift = random_state.randint(fs, tmax - fs)
                exp_phase_s = np.roll(exp_phase, shift)
                exp_phase_s *= amplitude
                MI_surr[s] = np.abs(np.mean(exp_phase_s))

            MI -= np.mean(MI_surr)
            MI /= np.std(MI_surr)

    elif method == 'tort':
        # Modulation index as in Tort 2010

        # mean amplitude distribution along phase bins
        n_bins = 18 + 2
        while n_bins > 0:
            n_bins -= 2
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            bin_indices = np.digitize(phase, phase_bins) - 1
            amplitude_dist = np.zeros(n_bins)
            for b in range(n_bins):
                selection = amplitude[bin_indices == b]
                if selection.size == 0:  # no sample in that bin
                    continue
                amplitude_dist[b] = np.mean(selection)
            if np.any(amplitude_dist == 0):
                continue

            # Kullback-Leibler divergence of the distribution vs uniform
            amplitude_dist /= np.sum(amplitude_dist)
            divergence_kl = np.sum(amplitude_dist *
                                   np.log(amplitude_dist * n_bins))

            MI = divergence_kl / np.log(n_bins)
            break

        if draw_phase:
            phase_bins = 0.5 * (phase_bins[:-1] + phase_bins[1:]) / np.pi * 180
            plt.plot(phase_bins, amplitude_dist, '.-')
            plt.plot(phase_bins, np.ones(n_bins) / n_bins, '--')
            plt.ylim((0, 2. / n_bins))
            plt.xlim((-180, 180))
            plt.ylabel('Normalized mean amplitude')
            plt.xlabel('Phase (in degree)')
            plt.title('Tort index: %.3f' % MI)

    else:
        raise(ValueError,
              "Unknown method for modulation index: Got '%s' instead "
              "of one in ('canolty', 'ozkurt', 'tort')" % method)

    return MI


def modulation_index(sig, fs,
                     low_fq_range=np.linspace(1.0, 10.0, 50),
                     low_fq_width=0.5,
                     high_fq_range=np.linspace(5.0, 150.0, 60),
                     high_fq_width=10.0,
                     method='tort',
                     n_surrogates=100,
                     draw='', save_name=None,
                     vmin=None, vmax=None,
                     return_filtered=False,
                     progress_bar=True,
                     draw_phase=False):
    """
    Compute the modulation index (MI) for Phase Amplitude Coupling (PAC).

    Parameters
    ----------
    sig           : one dimension signal
    fs            : sampling frequency
    low_fq_range  : low frequency range to compute the MI (phase signal)
    low_fq_width  : width of the band-pass filter
    high_fq_range : high frequency range to compute the MI (amplitude signal)
    high_fq_width : width of the band-pass filter
    method        : normalization method, in ('ozkurt', 'canolty', 'tort')
    n_surrogates  : number of surrogates computed in 'canolty's method
    draw          : if True, draw the comodulogram
    vmin, vmax    : if not None, it define the min/max value of the plot
    return_filtered: if True, also return the filtered signals
    draw_phase    : if True, plot the phase distribution in 'tort' index

    Return
    ------
    MI            : Modulation Index,
                    shape (len(low_fq_range), len(high_fq_range))
    """
    sig = sig.ravel()
    sig = crop_for_fast_hilbert(sig)
    tmax = sig.size

    # convert to numpy array
    low_fq_range = np.asarray(low_fq_range)
    high_fq_range = np.asarray(high_fq_range)

    # compute a number of band-pass filtered and Hilbert filtered signals
    filtered_high = multiple_band_pass(sig, fs, high_fq_range, high_fq_width)
    filtered_low = multiple_band_pass(sig, fs, low_fq_range, low_fq_width)

    MI = _modulation_index(filtered_low, filtered_high, method, fs,
                           n_surrogates, progress_bar, draw_phase)

    if draw:
        vmin = MI.min() if vmin is None else vmin
        vmax = MI.max() if vmax is None else vmax
        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(9, 9)
        ax_vert = plt.subplot(gs[:-2, :2])
        ax_hori = plt.subplot(gs[-2:, 2:-1])
        ax_main = plt.subplot(gs[:-2, 2:-1])
        ax_cbar = plt.subplot(gs[:-2, -1])

        extent = [low_fq_range[0], low_fq_range[-1],
                  high_fq_range[0], high_fq_range[-1]]
        cax = ax_main.imshow(MI.T, cmap=plt.cm.viridis,
                             aspect='auto', origin='lower', extent=extent,
                             interpolation='none', vmax=vmax, vmin=vmin)

        fig.colorbar(cax, cax=ax_cbar, ticks=np.linspace(vmin, vmax, 6))
        ax_hori.set_xlabel('Phase frequency (Hz) (w=%.1f)' % low_fq_width)
        ax_vert.set_ylabel('Amplitude frequency (Hz) (w=%.2f)' % high_fq_width)
        plt.suptitle('Phase Amplitude Coupling measured with Modulation Index'
                     ' (%s)' % method,
                     fontsize=14)

        ax_vert.plot(np.mean(MI.T, axis=1), high_fq_range)
        ax_vert.set_ylim(extent[2:])
        ax_hori.plot(low_fq_range, np.mean(MI.T, axis=0))
        ax_hori.set_xlim(extent[:2])

        if save_name is None:
            save_name = '%s_tmax%d_wlo%.2f_whi%.1f' % (method, tmax,
                                                       low_fq_width,
                                                       high_fq_width)
        fig.savefig(save_name + '.png')

    if return_filtered:
        return MI, filtered_low, filtered_high
    else:
        return MI
