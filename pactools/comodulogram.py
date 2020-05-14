import warnings

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interp2d

from .dar_model.base_dar import BaseDAR
from .dar_model.dar import DAR
from .dar_model.preprocess import multiple_extract_driver
from .utils.parallel import Parallel, delayed
from .utils.progress_bar import ProgressBar
from .utils.spectrum import Bicoherence, Coherence
from .utils.maths import norm, argmax_2d, next_power2
from .utils.validation import check_array, check_random_state
from .utils.validation import check_consistent_shape, check_is_fitted
from .utils.viz import add_colorbar
from .bandpass_filter import multiple_band_pass
from .mne_api import MaskIterator, _check_mne

N_BINS_TORT = 18

STANDARD_PAC_METRICS = ['ozkurt', 'canolty', 'tort', 'penny', 'vanwijk']
DAR_BASED_PAC_METRICS = ['duprelatour']
COHERENCE_PAC_METRICS = ['jiang', 'colgin']
BICOHERENCE_PAC_METRICS = ['sigl', 'nagashima', 'hagihira', 'bispectrum']

ALL_PAC_METRICS = (STANDARD_PAC_METRICS + DAR_BASED_PAC_METRICS +
                   COHERENCE_PAC_METRICS + BICOHERENCE_PAC_METRICS)


class Comodulogram(object):
    """An object to compute the comodulogram for phase-amplitude coupling.

    Parameters
    ----------
    fs : float
        Sampling frequency

    low_fq_range : array or list
        List of filtering frequencies (phase signal)

    high_fq_range : array or list or 'auto'
        List of filtering frequencies (amplitude signal)
        If 'auto', it uses np.linspace(max(low_fq_range), fs / 2.0, 40).

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    high_fq_width : float or 'auto'
        Bandwidth of the band-pass filter (amplitude signal)
        If 'auto', it uses 2 * max(low_fq_range).

    method : string or DAR instance
        Modulation index method:

        - String in ('ozkurt', 'canolty', 'tort', 'penny', ), for a PAC
            estimation based on filtering and using the Hilbert transform.
        - String in ('vanwijk', ) for a joint AAC and PAC estimation
            based on filtering and using the Hilbert transform.
        - String in ('sigl', 'nagashima', 'hagihira', 'bispectrum', ), for
            a PAC estimation based on the bicoherence.
        - String in ('colgin', ) for a PAC estimation
            and in ('jiang', ) for a PAC directionality estimation,
            based on filtering and computing coherence.
        - String in ('duprelatour', ) or a DAR instance, for a PAC estimation
            based on a driven autoregressive model.

    n_surrogates : int
        Number of surrogates computed for the z-score
        If n_surrogates <= 1, the z-score is not computed.

    vmin, vmax : float or None
        If not None, it define the min/max value of the plot.

    progress_bar : boolean
        If True, a progress bar is shown in stdout.

    ax_special : matplotlib.axes.Axes or None
        If not None, a special figure is drawn on it, depending on
        the PAC method used.

    minimum_shift : float
        Minimum time shift (in sec) for the surrogate analysis.

    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the surrogate analysis.

    coherence_params : dict
        Parameters for methods base on coherence or bicoherence.
        May contain:

        -block_length : int
            Block length
        -fft_length : int or None
            Length of the FFT
        -step : int or None
            Step between two blocks

        If the dictionary is empty, default values will be applied based on
        fs and low_fq_width, with 0.5 overlap windows and no zero-padding.

    extract_params : dict
        Parameters for DAR models driver extraction

    low_fq_width_2 : float
        Bandwidth of the band-pass filters centered on low_fq_range, for
        the amplitude signal. Used only with 'vanwijk' method.

    n_jobs : int
        Number of jobs to use in parallel computations.
        Recquires scikit-learn installed.

    Examples
    --------
    >>> from pactools.comodulogram import Comodulogram
    >>> c = Comodulogram(fs=200., low_fq_range=np.arange(2, 4, 0.2),
    ...                  low_fq_width=2.)
    >>> c.fit(signal_in)
    >>> c.plot()
    >>> comod_array = c.comod_
    """

    def __init__(self, fs, low_fq_range, low_fq_width=2., high_fq_range='auto',
                 high_fq_width='auto', method='tort', n_surrogates=0,
                 vmin=None, vmax=None, progress_bar=True, ax_special=None,
                 minimum_shift=1.0, random_state=None, coherence_params=dict(),
                 extract_params=dict(), low_fq_width_2=4.0, n_jobs=1):
        self.fs = fs
        self.low_fq_range = low_fq_range
        self.low_fq_width = low_fq_width
        self.high_fq_range = high_fq_range
        self.high_fq_width = high_fq_width
        self.method = method
        self.n_surrogates = n_surrogates
        self.vmin = vmin
        self.vmax = vmax
        self.progress_bar = progress_bar
        self.ax_special = ax_special
        self.minimum_shift = minimum_shift
        self.random_state = random_state
        self.coherence_params = coherence_params
        self.extract_params = extract_params
        self.low_fq_width_2 = low_fq_width_2
        self.n_jobs = n_jobs

    def _check_params(self):
        high_fq_range = self.high_fq_range
        if isinstance(high_fq_range, str) and high_fq_range == 'auto':
            self.high_fq_range = np.linspace(
                max(self.low_fq_range), self.fs / 2.0, 80)

        high_fq_width = self.high_fq_width
        if isinstance(high_fq_width, str) and high_fq_width == 'auto':
            self.high_fq_width = max(self.low_fq_range) * 2

        self.random_state = check_random_state(self.random_state)

        if isinstance(self.method, str):
            self.method = self.method.lower()

        self.fs = float(self.fs)

        self.low_fq_range = np.atleast_1d(self.low_fq_range)
        self.high_fq_range = np.atleast_1d(self.high_fq_range)

        if self.ax_special is not None:
            assert isinstance(self.ax_special, matplotlib.axes.Axes)
            if self.low_fq_range.size > 1:
                raise ValueError("ax_special can only be used if low_fq_range "
                                 "contains only one frequency.")

    def fit(self, low_sig, high_sig=None, mask=None):
        """Call fit to compute the comodulogram.

        Parameters
        ----------
        low_sig : array, shape (n_epochs, n_points)
            Input data for the phase signal

        high_sig : array or None, shape (n_epochs, n_points)
            Input data for the amplitude signal.
            If None, we use low_sig for both signals.

        mask : array or list of array or None, shape (n_epochs, n_points)
            The PAC is only evaluated where the mask is False.
            Masking is done after filtering and Hilbert transform.
            If the method computes the bicoherence, the mask has to be
            unidimensional (n_points, ) and the same mask is applied on all
            epochs.
            If a list or a MaskIterator is given, the filtering is done only
            once and the comodulogram is computed on each mask.


        Attributes
        ----------
        comod_ : array, shape (len(low_fq_range), len(high_fq_range))
            Comodulogram for each couple of frequencies.
            If a list of mask is given, it returns a array of shape
            (n_masks, len(low_fq_range), len(high_fq_range))

        surrogates_ : array, shape (n_surrogates, len(low_fq_range),
                      len(high_fq_range)) or None
            Comodulograms for each time shift of the surrogate analysis.
            If a list of mask is given, it returns a array of shape
            (n_masks, n_surrogates, len(low_fq_range), len(high_fq_range))
        """
        self._check_params()
        low_sig = check_array(low_sig)
        high_sig = check_array(high_sig, accept_none=True)
        check_consistent_shape(low_sig, high_sig)

        # check the masks
        multiple_masks = (isinstance(mask, list)
                          or isinstance(mask, MaskIterator)
                          or (isinstance(mask, np.ndarray) and mask.ndim == 3))
        if not multiple_masks:
            mask = [mask]
        if not isinstance(mask, MaskIterator):
            mask = [check_array(m, dtype=bool, accept_none=True) for m in mask]
        n_masks = len(mask)

        # pre-compute all the random time shifts
        n_epochs, n_points = low_sig.shape
        self.shifts_ = _get_shifts(self.random_state, n_points,
                                   self.minimum_shift, self.fs,
                                   self.n_surrogates)

        if self.method in STANDARD_PAC_METRICS:
            if high_sig is None:
                high_sig = low_sig

            if self.progress_bar:
                self.progress_bar = ProgressBar(
                    'comodulogram: %s' % self.method,
                    max_value=self.low_fq_range.size * n_masks)

            # compute a number of band-pass filtered signals
            filtered_high = multiple_band_pass(
                high_sig, self.fs, self.high_fq_range, self.high_fq_width)
            filtered_low = multiple_band_pass(
                low_sig, self.fs, self.low_fq_range, self.low_fq_width)
            if self.method == 'vanwijk':
                filtered_low_2 = multiple_band_pass(
                    low_sig, self.fs, self.low_fq_range, self.low_fq_width_2)
            else:
                filtered_low_2 = None

            all_results = []
            for this_mask in mask:
                results = _comodulogram(self, filtered_low, filtered_high,
                                        this_mask, filtered_low_2)
                all_results.append(results)

        elif self.method in COHERENCE_PAC_METRICS:
            if high_sig is None:
                high_sig = low_sig

            if self.progress_bar:
                self.progress_bar = ProgressBar(
                    'coherence: %s' % self.method,
                    max_value=n_masks * len(self.shifts_))

            # compute a number of band-pass filtered signals
            filtered_high = multiple_band_pass(
                high_sig, self.fs, self.high_fq_range, self.high_fq_width)

            all_results = []
            for this_mask in mask:
                results = _coherence(self, low_sig, filtered_high, this_mask)
                all_results.append(results)

        # compute PAC with the bispectrum/bicoherence
        elif self.method in BICOHERENCE_PAC_METRICS:
            if high_sig is not None:
                raise ValueError(
                    "Impossible to use a bicoherence method (%s) on two "
                    "signals, please try another method." % self.method)
            if self.n_surrogates > 1:
                raise NotImplementedError(
                    "Surrogate analysis with a bicoherence method (%s) "
                    "is not implemented." % self.method)

            if self.progress_bar:
                self.progress_bar = ProgressBar(
                    'bicoherence: %s' % self.method, max_value=n_masks)

            all_results = []
            for this_mask in mask:
                results = _bicoherence(self, sig=low_sig, mask=this_mask)
                all_results.append([results])
                if self.progress_bar:
                    self.progress_bar.update_with_increment_value(1)

        elif isinstance(self.method,
                        BaseDAR) or self.method in DAR_BASED_PAC_METRICS:

            all_results = _driven_comodulogram(self, low_sig=low_sig,
                                               high_sig=high_sig, mask=mask)
        else:
            raise ValueError('unknown method: %s' % self.method)

        # remove very small values
        all_results = np.asarray(all_results)
        all_results[np.abs(all_results) < 10 * np.finfo(np.float64).eps] = 0

        self.comod_ = all_results[:, 0, :, :]
        self.surrogates_ = all_results[:, 1:, :, :]
        if not multiple_masks:
            self.comod_ = self.comod_[0]
            self.surrogates_ = self.surrogates_[0]

        return self

    @property
    def comod_z_score_(self):
        """Compute the z-score based on the comodulogram and the surrogates

        Returns
        -------
        comod_z_score_: array, shape (len(low_fq_range), len(high_fq_range))
            If self.comod_ is an array of n_masks comodulograms, so does this
            array: shape (n_masks, len(low_fq_range), len(high_fq_range))
        """
        check_is_fitted(self, 'surrogates_')

        comod_ = self.comod_.copy()
        surrogates_ = self.surrogates_

        ndim = surrogates_.ndim
        if ndim == 3:
            surrogates_ = surrogates_[None, ...]
            comod_ = comod_[None, ...]
        # n_masks, n_surrogates, n_low, n_high = self.surrogates_.shape

        if surrogates_.shape[1] > 2:
            comod_z_score = comod_
            comod_z_score -= np.mean(surrogates_, axis=1)
            comod_z_score /= np.std(surrogates_, axis=1)
            if ndim == 3:
                comod_z_score = comod_z_score[0]
            return comod_z_score
        else:
            raise ValueError(
                "Impossible to compute comod_z_score_ since the surrogate "
                "comodulograms were not computed. Try to refit the "
                "estimator with n_surrogates > 1.")

    @property
    def surrogate_max_(self):
        """Compute the distribution of maxima of the surrogates comodulograms

        Returns
        -------
        surrogate_max_: array, shape (n_surrogates, )
            If self.comod_ is an array of n_masks comodulograms, the shape will
            be: shape (n_masks, n_surrogates)
        """
        check_is_fitted(self, 'surrogates_')
        surrogates_ = self.surrogates_

        ndim = surrogates_.ndim
        if ndim == 3:
            surrogates_ = surrogates_[None, ...]
        # n_masks, n_surrogates, n_low, n_high = self.surrogates_.shape

        if surrogates_.shape[1] > 2:
            n_masks, n_surrogates, n_low, n_high = surrogates_.shape
            tmp = surrogates_.reshape(n_masks, n_surrogates, -1)
            surrogate_max = np.max(tmp, axis=2)
            if ndim == 3:
                surrogate_max = surrogate_max[0]
            return surrogate_max
        else:
            raise ValueError(
                "Impossible to compute comod_z_score_ since the surrogate "
                "comodulograms were not computed. Try to refit the "
                "estimator with n_surrogates > 1.")

    def plot(self, titles=None, axs=None, cmap=None, vmin=None, vmax=None,
             unit='', cbar=True, label=True, contour_level=None,
             contour_method='comod_max', tight_layout=True):
        """
        Plot the comodulograms computed during the fit

        titles : list of string or None
            List of titles for each comodulogram

        axs : list or array of matplotlib.axes.Axes
            Axes where the comodulograms are drawn. If None, a new figure is
            created. Typical use is: fig, axs = plt.subplots(3, 4)

        cmap : colormap or None
            Colormap used in the plot. If None, it uses 'viridis' colormap.

        vmin, vmax : float or None
            If not None, they define the min/max value of the plot, else
            they are set to (0, comodulograms.max()).

        unit : string (default: '')
            Unit of the comodulogram

        cbar : boolean
            Display colorbar or not

        label : boolean
            Display labels or not

        contour_level : None or float
            If not None, contours will be added around values where the
            significance level is above contour value. The significance level
            is computed only when n_surrogates >= 2.

        contour_method : str in ('comod_max', 'z_score'), (default:'comod_max')
            Select the method used to compute the significance level.
                - 'comod_max': Compute the maximum of each surrogate
                    comodulogram, then estimate a threshold that correspond to
                    a p-value specified in contour_level.
                - 'z_score': Compute the z-score for each couple of frequencies
                    This method might suffer from multiple testing issues.

        tight_layout : boolean
            Use tight_layout or not

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure object
        """
        check_is_fitted(self, 'comod_')
        comod_ = self.comod_
        # resize for generality
        if comod_.ndim == 2:
            comod_ = comod_[None, :, :]
        n_comod, n_low_fq, n_high_fq = comod_.shape

        # generate a new figure if no axes is provided
        if axs is None:
            n_lines = int(np.sqrt(n_comod))
            n_columns = int(np.ceil(n_comod / float(n_lines)))
            figsize = (4 * n_columns, 3 * n_lines)
            fig, axs = plt.subplots(n_lines, n_columns, figsize=figsize)
        else:
            fig = axs[0].figure
        axs = np.array(axs).ravel()

        # Get auto values for vmin, vmax
        if vmin is None and vmax is None:
            vmin = min(0, comod_.min())
            vmax = max(0, comod_.max())
            if vmin < 0 and vmax > 0:
                vmax = max(vmax, -vmin)
                vmin = -vmax
                if cmap is None:
                    cmap = plt.get_cmap('RdBu_r')

        if cmap is None:
            cmap = plt.get_cmap('viridis')

        extent = [
            self.low_fq_range[0],
            self.low_fq_range[-1],
            self.high_fq_range[0],
            self.high_fq_range[-1],
        ]

        # plot the image
        for i in range(n_comod):
            cax = axs[i].imshow(comod_[i].T, cmap=cmap, vmin=vmin, vmax=vmax,
                                aspect='auto', origin='lower', extent=extent,
                                interpolation='none')

        # optional parameter
        if titles is not None:
            for i in range(n_comod):
                axs[i].set_title(titles[i], fontsize=12)

        if label:
            axs[-1].set_xlabel('Driver frequency (Hz)')
            axs[0].set_ylabel('Signal frequency (Hz)')

        if tight_layout:
            fig.tight_layout()

        if cbar:
            # plot the colorbar once
            ax = axs[0] if len(axs) == 1 else None
            add_colorbar(fig, cax, vmin, vmax, unit=unit, ax=ax)

        # plot the contours
        if contour_level is not None:
            surrogate_max_ = self.surrogate_max_
            comod_z_score_ = self.comod_z_score_
            # resize for generality
            if surrogate_max_.ndim == 1:
                surrogate_max_ = surrogate_max_[None, :]
                comod_z_score_ = comod_z_score_[None, :, :]

            for i in range(n_comod):
                if contour_method == 'comod_max':
                    p_values = np.atleast_1d(contour_level)
                    percentiles = 100. * (1 - p_values)
                    levels = np.atleast_1d(
                        np.percentile(surrogate_max_[i], percentiles))

                    axs[i].contour(comod_[i].T, levels=levels, colors='w',
                                   origin='lower', extent=extent)
                elif contour_method == 'z_score':
                    levels = np.atleast_1d(contour_level)
                    axs[i].contour(comod_z_score_[i].T, levels=levels,
                                   colors='w', origin='lower', extent=extent)
                else:
                    raise ValueError(
                        "contour_method has to be one of ('comod_max', "
                        "'z_score'), got %s" % (contour_method, ))

        return fig

    def get_maximum_pac(self):
        """Get maximum PAC value in a comodulogram.

        Returns
        -------
        low_fq : float or array, shape (n_masks, )
            Low frequency of maximum PAC

        high_fq : float or array, shape (n_masks, )
            High frequency of maximum PAC

        pac_value : float or array, shape (n_masks, )
            Maximum PAC value
        """
        check_is_fitted(self, 'comod_')
        # only one comodulogram
        return_array = True
        if self.comod_.ndim == 2:
            self.comod_ = self.comod_[None, :, :]
            return_array = False

        # check that the sizes match
        n_masks, n_low, n_high = self.comod_.shape

        # compute the maximum of the comodulogram, and get the frequencies
        max_pac_value = np.zeros(n_masks)
        low_fq = np.zeros(n_masks)
        high_fq = np.zeros(n_masks)
        for k, comodulogram in enumerate(self.comod_):
            i, j = argmax_2d(comodulogram)
            max_pac_value[k] = comodulogram[i, j]

            low_fq[k] = self.low_fq_range[i]
            high_fq[k] = self.high_fq_range[j]

        # return arrays or floats
        if return_array:
            return low_fq, high_fq, max_pac_value
        else:
            return low_fq[0], high_fq[0], max_pac_value[0]

    def save(self, fname, overwrite=False):
        """
        Save a comodulogram object on disk with h5py.

        Parameters
        ----------
        fname : string
            Filename to use.
        overwrite : boolean
            If True, overwrite file if it exists.

        Examples
        --------
        >>> from pactools.comodulogram import Comodulgram, read_comodulogram
        >>> est = Comodulogram(fs=1000., low_fq_range=[1, 2, 3])
        >>> fname = '/my_filename.hdf5'
        >>> est.save(fname)
        >>> est2 = read_comodulogram(fname)
        """
        mne = _check_mne('Comodulogram.save')
        write_hdf5 = mne.externals.h5io.write_hdf5

        save_vars = {k: v for k, v in vars(self).items()}
        if 'random_state' in save_vars:
            rs = save_vars['random_state']
            if type(rs) != np.int:
                save_vars['random_state'] = 'mtrand'
                save_vars['random_state_state_'] = rs.__getstate__()
            if self.progress_bar:
                save_vars['progress_bar'] = True
        write_hdf5(fname, save_vars, title='comodulogram', overwrite=overwrite,
                   slash='replace')


def _comodulogram(estimator, filtered_low, filtered_high, mask,
                  filtered_low_2):
    """
    Helper function to compute the comodulogram.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    # The modulation index is only computed where mask is True
    if mask is not None:
        filtered_low = filtered_low[:, ~mask]
        filtered_high = filtered_high[:, ~mask]
        if estimator.method == 'vanwijk':
            filtered_low_2 = filtered_low_2[:, ~mask]
    else:
        filtered_low = filtered_low.reshape(filtered_low.shape[0], -1)
        filtered_high = filtered_high.reshape(filtered_high.shape[0], -1)
        if estimator.method == 'vanwijk':
            filtered_low_2 = filtered_low_2.reshape(filtered_low_2.shape[0],
                                                    -1)

    n_low, _ = filtered_low.shape
    n_high, _ = filtered_high.shape
    n_shifts = estimator.shifts_.size

    # phase of the low frequency signals
    for i in range(n_low):
        filtered_low[i] = np.angle(filtered_low[i])
    filtered_low = np.real(filtered_low)

    # amplitude of the high frequency signals
    filtered_high = np.real(np.abs(filtered_high))
    norm_a = np.zeros(n_high)
    if estimator.method == 'ozkurt':
        for j in range(n_high):
            norm_a[j] = norm(filtered_high[j])

    # amplitude of the low frequency signals
    if estimator.method == 'vanwijk':
        for i in range(n_low):
            filtered_low_2[i] = np.abs(filtered_low_2[i])
        filtered_low_2 = np.real(filtered_low_2)

    # Calculate the modulation index for each couple
    comod_list = np.zeros((n_shifts, n_low, n_high))
    for i in range(n_low):
        # preproces the phase array
        if estimator.method == 'tort':
            n_bins = N_BINS_TORT
            # make sure edge case for right edge is included
            eps = np.finfo(filtered_low[i].dtype).eps * 2
            phase_bins = np.linspace(-np.pi, np.pi + eps, n_bins + 1)
            # get the indices of the bins to which each value in input belongs
            phase_preprocessed = np.digitize(filtered_low[i], phase_bins) - 1
        elif estimator.method == 'penny':
            phase_preprocessed = np.c_[np.ones_like(filtered_low[i]),
                                       np.cos(filtered_low[i]),
                                       np.sin(filtered_low[i])]
        elif estimator.method == 'vanwijk':
            phase_preprocessed = np.c_[np.ones_like(filtered_low[i]),
                                       np.cos(filtered_low[i]),
                                       np.sin(filtered_low[i]),
                                       filtered_low_2[i]]  # yapf: disable
        elif estimator.method in ('canolty', 'ozkurt'):
            phase_preprocessed = np.exp(1j * filtered_low[i])
        else:
            raise ValueError('Unknown method %s.' % estimator.method)

        delayed_func = delayed(_loop_over_shifts)
        mi_list = Parallel(n_jobs=estimator.n_jobs)(delayed_func(
            _one_modulation_index, estimator.shifts_,
            amplitude=filtered_high[j], phase_preprocessed=phase_preprocessed,
            norm_a=norm_a[j], method=estimator.method,
            ax_special=estimator.ax_special) for j in range(n_high))

        mi_list = np.array(mi_list).reshape(n_high, n_shifts).T
        comod_list[:, i, :] = mi_list

        if estimator.progress_bar:
            estimator.progress_bar.update_with_increment_value(1)

    return comod_list


def _loop_over_shifts(func, shifts, **kwargs):
    """Helper to loop over shifts inside a single job"""
    return [func(shift=sh, **kwargs) for sh in shifts]


def _one_modulation_index(amplitude, phase_preprocessed, norm_a, method, shift,
                          ax_special):
    """
    Compute one modulation index.
    Used by PAC method in STANDARD_PAC_METRICS.
    """
    # shift for the surrogate analysis
    if shift != 0:
        phase_preprocessed = np.roll(phase_preprocessed, shift)

    # Modulation index as in [Ozkurt & al 2011]
    if method == 'ozkurt':
        MI = np.abs(np.mean(amplitude * phase_preprocessed))
        MI *= np.sqrt(amplitude.size) / norm_a

    # Generalized linear models as in [Penny & al 2008] or [van Wijk & al 2015]
    elif method in ('penny', 'vanwijk'):
        # solve a linear regression problem:
        # amplitude = np.dot(phase_preprocessed) * beta
        PtP = np.dot(phase_preprocessed.T, phase_preprocessed)
        PtA = np.dot(phase_preprocessed.T, amplitude[:, None])
        beta = np.linalg.solve(PtP, PtA)
        residual = amplitude - np.dot(phase_preprocessed, beta).ravel()
        variance_amplitude = np.var(amplitude)
        variance_residual = np.var(residual)
        MI = (variance_amplitude - variance_residual) / variance_amplitude

    # Modulation index as in [Canolty & al 2006]
    elif method == 'canolty':
        z_array = amplitude * phase_preprocessed
        MI = np.abs(np.mean(z_array))

        if ax_special is not None and shift == 0:
            ax_special.plot(np.real(z_array), np.imag(z_array))
            ax_special.set_ylabel('Imaginary part of z(t)')
            ax_special.set_xlabel('Real part of z(t)')
            ax_special.set_title("Canolty's modulation index: %.3f" % MI)
            ax_special.grid('on')

    # Modulation index as in [Tort & al 2010]
    elif method == 'tort':
        # mean amplitude distribution along phase bins
        n_bins = N_BINS_TORT
        amplitude_dist = np.ones(n_bins)  # default is 1 to avoid log(0)
        for b in np.unique(phase_preprocessed):
            selection = amplitude[phase_preprocessed == b]
            amplitude_dist[b] = np.mean(selection)

        # Kullback-Leibler divergence of the distribution vs uniform
        amplitude_dist /= np.sum(amplitude_dist)
        divergence_kl = np.sum(
            amplitude_dist * np.log(amplitude_dist * n_bins))

        MI = divergence_kl / np.log(n_bins)

        if ax_special is not None and shift == 0:
            phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
            phase_bins = 0.5 * (phase_bins[:-1] + phase_bins[1:]) / np.pi * 180
            ax_special.plot(phase_bins, amplitude_dist, '.-')
            ax_special.plot(phase_bins, np.ones(n_bins) / n_bins, '--')
            ax_special.set_ylim((0, 2. / n_bins))
            ax_special.set_xlim((-180, 180))
            ax_special.set_ylabel('Normalized mean amplitude')
            ax_special.set_xlabel('Phase (in degree)')
            ax_special.set_title("Tort's modulation index: %.3f" % MI)

    else:
        raise ValueError("Unknown method: %s" % (method, ))

    return MI


def _same_mask_on_all_epochs(sig, mask, method):
    """
    PAC metrics based on coherence or bicoherence,
    the same mask is applied on all epochs.
    """
    mask = np.squeeze(mask)
    if mask.ndim > 1:
        warnings.warn("For coherence methods (e.g. %s) the mask has "
                      "to be unidimensional, and the same mask is "
                      "applied on all epochs. Got shape %s, so only the "
                      "first row of the mask is used." %
                      (method, mask.shape, ), UserWarning)
        mask = mask[0, :]
    sig = sig[..., ~mask]
    return sig


def _bicoherence(estimator, sig, mask):
    """
    Helper function for the comodulogram.
    Used by PAC method in BICOHERENCE_PAC_METRICS.
    """
    # The modulation index is only computed where mask is True
    if mask is not None:
        sig = _same_mask_on_all_epochs(sig, mask, estimator.method)

    n_epochs, n_points = sig.shape

    coherence_params = _define_default_coherence_params(
        estimator.fs, estimator.low_fq_width, estimator.method,
        **estimator.coherence_params)

    model = Bicoherence(**coherence_params)
    bicoh = model.fit(sigs=sig, method=estimator.method)

    # remove the redundant part
    n_freq = bicoh.shape[0]
    np.flipud(bicoh)[np.triu_indices(n_freq, 1)] = 0
    bicoh[np.triu_indices(n_freq, 1)] = 0

    frequencies = np.linspace(0, estimator.fs / 2., n_freq)
    comod = _interpolate(frequencies, frequencies, bicoh,
                         estimator.high_fq_range, estimator.low_fq_range)

    return comod


def _define_default_coherence_params(fs, low_fq_width, method, **user_params):
    """
    Define default values for Coherence and Bicoherence classes,
    if not defined in user_params dictionary.
    """

    # the FFT length is chosen to have a frequency resolution of low_fq_width
    fft_length = fs / low_fq_width
    # but it is faster if it is a power of 2
    fft_length = 2 ** int(np.ceil(np.log2(fft_length)))

    # smoothing for bicoherence methods
    if method in BICOHERENCE_PAC_METRICS:
        fft_length /= 4
    # not smoothed for because we convolve after
    if method == 'jiang':
        fft_length *= 2

    # the block length is chosen to avoid zero-padding
    block_length = fft_length

    if 'block_length' not in user_params and 'fft_length' not in user_params:
        user_params['block_length'] = block_length
        user_params['fft_length'] = fft_length
    elif 'block_length' in user_params and 'fft_length' not in user_params:
        user_params['fft_length'] = user_params['block_length']
    elif 'block_length' not in user_params and 'fft_length' in user_params:
        user_params['block_length'] = user_params['fft_length']

    if 'fs' not in user_params:
        user_params['fs'] = fs
    if 'step' not in user_params:
        user_params['step'] = None

    return user_params


def _coherence(estimator, low_sig, filtered_high, mask):
    """
    Helper function to compute the comodulogram.
    Used by PAC method in COHERENCE_PAC_METRICS.
    """
    if mask is not None:
        low_sig = _same_mask_on_all_epochs(low_sig, mask, estimator.method)
        filtered_high = _same_mask_on_all_epochs(filtered_high, mask,
                                                 estimator.method)

    # amplitude of the high frequency signals
    filtered_high = np.real(np.abs(filtered_high))

    coherence_params = _define_default_coherence_params(
        estimator.fs, estimator.low_fq_width, estimator.method,
        **estimator.coherence_params)

    delayed_func = delayed(_one_coherence_modulation_index)
    generator = (delayed_func(
        shift=sh, fs=estimator.fs, low_sig=low_sig,
        filtered_high=filtered_high, method=estimator.method,
        low_fq_range=estimator.low_fq_range, coherence_params=coherence_params)
        for sh in estimator.shifts_)
    if estimator.progress_bar:
        generator = estimator.progress_bar(generator)

    comod_list = Parallel(n_jobs=estimator.n_jobs)(generator)
    comod_list = np.array(comod_list)

    return comod_list


def _one_coherence_modulation_index(fs, low_sig, filtered_high, method,
                                    low_fq_range, coherence_params, shift):
    """
    Compute one modulation index.
    Used by PAC method in COHERENCE_PAC_METRICS.
    """
    if shift != 0:
        low_sig = np.roll(low_sig, shift)

    # the actual frequency resolution is computed here
    delta_freq = fs / coherence_params['fft_length']

    model = Coherence(**coherence_params)
    coherence = model.fit(low_sig[None, :, :], filtered_high)[0]
    n_high, n_freq = coherence.shape
    frequencies = np.linspace(0, fs / 2., n_freq)

    # Coherence as in [Colgin & al 2009]
    if method == 'colgin':
        coherence = np.real(np.abs(coherence))

        comod = _interpolate(
            np.arange(n_high), frequencies, coherence,
            np.arange(n_high), low_fq_range)

    # Phase slope index as in [Jiang & al 2015]
    elif method == 'jiang':
        product = coherence[:, 1:] * np.conjugate(coherence[:, :-1])

        # we use a kernel of (ker * 2) with respect to the product,
        # i.e. a kernel of (ker * 2 +1) with respect to the coherence.
        ker = 2
        kernel = np.ones(2 * ker) / (2 * ker)
        phase_slope_index = np.zeros((n_high, n_freq - (2 * ker)),
                                     dtype=np.complex128)
        for i in range(n_high):
            phase_slope_index[i] = np.convolve(product[i], kernel, 'valid')
        phase_slope_index = np.imag(phase_slope_index)
        frequencies = frequencies[ker:-ker]

        # transform the phase slope index into an approximated delay
        delay = phase_slope_index / (2. * np.pi * delta_freq)

        comod = _interpolate(
            np.arange(n_high), frequencies, delay,
            np.arange(n_high), low_fq_range)

    else:
        raise ValueError('Unknown method %s' % (method, ))

    return comod


def _interpolate(x1, y1, z1, x2, y2):
    """Helper to interpolate in 1d or 2d

    We interpolate to get the same shape than with other methods.
    """
    if x1.size > 1 and y1.size > 1:
        func = interp2d(x1, y1, z1.T, kind='linear', bounds_error=False)
        z2 = func(x2, y2)
    elif x1.size == 1 and y1.size > 1:
        func = interp1d(y1, z1.ravel(), kind='linear', bounds_error=False)
        z2 = func(y2)
    elif y1.size == 1 and x1.size > 1:
        func = interp1d(x1, z1.ravel(), kind='linear', bounds_error=False)
        z2 = func(x2)
    else:
        raise ValueError("Can't interpolate a scalar.")

    # interp2d is not intuitive and return this shape:
    z2.shape = (y2.size, x2.size)
    return z2


def _driven_comodulogram(estimator, low_sig, high_sig, mask):
    """
    Helper function for the comodulogram.
    Used by PAC method in DAR_BASED_PAC_METRICS.
    """
    model = estimator.method
    if model == 'duprelatour':
        model = DAR(ordar=10, ordriv=1)

    n_epochs = low_sig.shape[0]
    if high_sig is None:
        sigs = low_sig
    else:
        # hack to call only once extract
        high_sig = np.atleast_2d(high_sig)
        sigs = np.r_[high_sig, low_sig]

    generator = (
        delayed(_driven_comodulogram_column)(estimator, filtered_signals,
                                             high_sig, mask, n_epochs)
        for filtered_signals in multiple_extract_driver(
            sigs=sigs, fs=estimator.fs, bandwidth=estimator.low_fq_width,
            frequency_range=estimator.low_fq_range,
            random_state=estimator.random_state, **estimator.extract_params))

    if estimator.progress_bar:
        bar = ProgressBar(max_value=len(estimator.low_fq_range) * len(mask),
                          title='comodulogram: %s' %
                          (model.get_title(name=True), ))
        generator = bar(generator)

    results = Parallel(n_jobs=estimator.n_jobs)(generator)

    if estimator.progress_bar:
        bar.update(cur_value=bar.max_value)

    all_results = np.array(results)
    # From: n_low_fq * n_masks * n_shifts * n_high_fq
    # Into: n_masks * n_shifts * n_low_fq * n_high_fq
    all_results = np.swapaxes(all_results, 0, 2)
    all_results = np.swapaxes(all_results, 0, 1)
    assert all_results.shape[0] == len(mask)
    assert all_results.shape[1] == len(estimator.shifts_)
    assert all_results.shape[2] == estimator.low_fq_range.size

    return all_results


def _driven_comodulogram_column(estimator, filtered_signals, high_sig, mask,
                                n_epochs):
    extract_complex = estimator.extract_params.get('extract_complex', True)

    model = estimator.method
    if model == 'duprelatour':
        model = DAR(ordar=10, ordriv=1)

    if extract_complex:
        filtered_low, filtered_high, filtered_low_imag = filtered_signals
    else:
        filtered_low, filtered_high = filtered_signals

    sigdriv_imag = None
    if high_sig is None:
        sigin = np.array(filtered_high)
        sigdriv = np.array(filtered_low)
        if extract_complex:
            sigdriv_imag = np.array(filtered_low_imag)
    else:
        sigin = np.array(filtered_high[:n_epochs])
        sigdriv = np.array(filtered_low[n_epochs:])
        if extract_complex:
            sigdriv_imag = np.array(filtered_low_imag[n_epochs:])

    sigin /= np.std(sigin)

    results = []
    for i_mask, this_mask in enumerate(mask):
        comod_list = []
        for sh in estimator.shifts_:
            comod = _one_driven_modulation_index(
                shift=sh, fs=estimator.fs, sigin=sigin, sigdriv=sigdriv,
                sigdriv_imag=sigdriv_imag, model=model, mask=this_mask,
                high_fq_range=estimator.high_fq_range,
                ax_special=estimator.ax_special)
            comod_list.append(comod)
        results.append(np.array(comod_list))

    return results


def _one_driven_modulation_index(fs, sigin, sigdriv, sigdriv_imag, model, mask,
                                 high_fq_range, ax_special, shift):
    """
    Compute one modulation index.
    Used by PAC method in DAR_BASED_PAC_METRICS.
    """
    # shift for the surrogate analysis
    if shift != 0:
        sigdriv = np.roll(sigdriv, shift)

    train_weights = ~mask if mask is not None else None

    # fit the model DAR on the data
    model.fit(fs=fs, sigin=sigin, sigdriv=sigdriv, sigdriv_imag=sigdriv_imag,
              train_weights=train_weights)

    # estimate the length of the padding for the FFT
    if len(high_fq_range) > 1:
        delta_f = np.diff(high_fq_range).mean()
        n_fft = next_power2(fs / delta_f)
    else:
        n_fft = 1024
    # get PSD difference
    spec, _, _, _ = model._amplitude_frequency(n_fft=n_fft)

    # KL divergence for each phase, as in [Tort & al 2010]
    n_freq, n_phases = spec.shape
    spec = 10. ** (spec / 20.)
    spec = spec / np.sum(spec, axis=1)[:, None]
    spec_diff = np.sum(spec * np.log(spec * n_phases), axis=1)
    spec_diff /= np.log(n_phases)

    # crop the spectrum to high_fq_range
    frequencies = np.linspace(0, fs // 2, spec_diff.size)
    spec_diff = np.interp(high_fq_range, frequencies, spec_diff)

    if ax_special is not None and shift == 0:
        model.plot(frange=[high_fq_range[0], high_fq_range[-1]], ax=ax_special)

    return spec_diff


def _get_shifts(random_state, n_points, minimum_shift, fs, n_surrogates):
    """Compute the shifts for the surrogate analysis"""
    n_iterations = max(1, 1 + n_surrogates)

    n_minimum_shift = max(1, int(fs * minimum_shift))
    # shift at least minimum_shift seconds, i.e. n_minimum_shift points
    if n_iterations > 1:
        if n_points - n_minimum_shift < n_minimum_shift:
            raise ValueError("The minimum shift is longer than half the "
                             "visible data.")

        shifts = random_state.randint(
            n_minimum_shift, n_points - n_minimum_shift, size=n_iterations)
    else:
        shifts = np.array([0])

    # the first has no shift since this is for the initial computation
    shifts[0] = 0

    return shifts


def read_comodulogram(fname):
    mne = _check_mne('read_comodulogram')
    read_hdf5 = mne.externals.h5io.read_hdf5

    data = read_hdf5(fname, 'comodulogram', slash='replace')
    init_params = {k: v for k, v in data.items() if not k.endswith('_')}
    if 'random_state' in init_params:
        rs = init_params['random_state']
        if rs == 'mtrand':
            rs_state = data['random_state_state_']
            del data['random_state_state_']
            rs = np.random.RandomState()
            rs.set_state(rs_state)
            init_params['random_state'] = rs
    attrs = {k: v for k, v in data.items() if k.endswith('_')}
    inst = Comodulogram(**init_params)
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst
