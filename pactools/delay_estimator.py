import numpy as np
import matplotlib.pyplot as plt

from .utils.validation import check_consistent_shape, check_array
from .utils.validation import check_is_fitted

from .dar_model import extract_driver
from .utils.progress_bar import ProgressBar
from .utils.viz import SEABORN_PALETTES


class DelayEstimator(object):
    """Estimate the optimal delay between the two components in PAC

    In phase-amplitude coupling (PAC), the slow oscillation and the fast
    oscillations may be shifted in time with a constant temporal delay. This
    estimator compute the optimal delay, based on the maximum likelihood of DAR
    models.

    Parameters
    ----------
    fs : float
        Sampling frequency

    dar_model : DAR instance
        DAR model used to fit the signal

    low_fq : float
        Filtering frequency (phase signal)

    low_fq_width : float
        Bandwidth of the band-pass filter (phase signal)

    max_delay : float or 'auto'
        The delay grid will range from -max_delay to max_delay.
        If 'auto', it uses 0.5 / low_fq.

    refit : boolean, default True
        If True, the model will be refitted with the best delay obtained

    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the surrogate analysis.

    References
    ----------
    Dupre la Tour et al. (2017). Non-linear Auto-Regressive Models for
    Cross-Frequency Coupling in Neural Time Series. bioRxiv, 159731.
    """

    def __init__(self, fs, dar_model, low_fq, low_fq_width, max_delay='auto',
                 refit=True, random_state=None):
        self.fs = fs
        self.dar_model = dar_model
        self.low_fq = low_fq
        self.low_fq_width = low_fq_width
        self.max_delay = max_delay
        self.refit = refit
        self.random_state = random_state

    def fit(self, low_sig, high_sig=None, mask=None):
        """
        Compute peak-locked time-averaged and time-frequency representations.

        Parameters
        ----------
        low_sig : array, shape (n_epochs, n_points)
            Input data for the phase signal

        high_sig : array or None, shape (n_epochs, n_points)
            Input data for the amplitude signal.
            If None, we use low_sig for both signals

        mask : array or None, shape (n_epochs, n_points)
            The model is only fitted where the mask is False.
            Masking is done after filtering, and is not delayed.

        Attributes
        ----------
        neg_log_likelihood_ : array, shape (n_delays, )
            Negative log-likelihood of dar_model, fitted on a grid of delays
            self.delays_ms_

        delays_ms_ : array, shape (n_delays, )
            Temporal delays (in ms), corresponding to self.neg_log_likelihood_

        best_delay_ms_ : float
            Temporal delay corresponding to the minimum negative log-likelihood

        best_model_ : fitted DAR instance
            If refit is True, the model is refitted with the best delay
        """
        self.low_sig = check_array(low_sig)
        self.high_sig = check_array(high_sig, accept_none=True)
        if self.high_sig is None:
            self.high_sig = self.low_sig

        self.mask = check_array(mask, accept_none=True)
        check_consistent_shape(self.low_sig, self.high_sig, self.mask)

        model = self.dar_model

        # window decay of sigdriv for continuity after np.roll
        n_decay = max(int(0.5 * self.fs / self.low_fq), 5)
        window = np.blackman(n_decay * 2 - 1)[:n_decay]
        self.low_sig = self.low_sig.copy()  # copy to avoid modifying original
        self.low_sig[:, :n_decay] *= window
        self.low_sig[:, -n_decay:] *= window[::-1]

        sigdriv, sigin, sigdriv_imag = extract_driver(
            self.low_sig, self.fs, self.low_fq, bandwidth=self.low_fq_width,
            fill=2, random_state=self.random_state)
        if self.high_sig is not self.low_sig:
            _, sigin, _ = extract_driver(self.high_sig, self.fs, self.low_fq,
                                         bandwidth=self.low_fq_width, fill=2,
                                         random_state=self.random_state)

        if self.max_delay == 'auto':
            max_delay_point = int(0.5 / self.low_fq * self.fs)
        else:
            max_delay_point = int(self.max_delay * self.fs)

        # delay in time points
        delays_point = np.arange(max_delay_point + 1)
        delays_point = np.r_[-delays_point[:0:-1], delays_point]
        bar = ProgressBar(title='delays', max_value=len(delays_point))
        self.delays_ms_ = delays_point / self.fs * 1000.

        train_weights = (1. - self.mask) if self.mask is not None else None

        neg_log_likelihood = np.zeros(len(delays_point))
        for i_delay, delay in enumerate(delays_point):
            # add delay
            sigdriv_ = np.roll(sigdriv, delay, axis=1)
            sigdriv_imag_ = np.roll(sigdriv_imag, delay, axis=1)

            # fit the model direct
            model.fit(sigin=sigin, sigdriv=sigdriv_,
                      sigdriv_imag=sigdriv_imag_, fs=self.fs,
                      train_weights=train_weights)
            neg_log_likelihood[i_delay] = model.get_criterion('-logl')
            # fit the model reverted
            model.fit(sigin=sigin[..., ::-1], sigdriv=sigdriv_[..., ::-1],
                      sigdriv_imag=sigdriv_imag_[..., ::-1], fs=self.fs,
                      train_weights=train_weights)
            neg_log_likelihood[i_delay] += model.get_criterion('-logl')

            bar.update(i_delay + 1)
        bar.close()
        self.neg_log_likelihood_ = neg_log_likelihood

        # compute the best delay
        i_best = np.nanargmin(neg_log_likelihood)
        self.best_delay_ms_ = self.delays_ms_[i_best]

        # refit the model with the best delay
        if self.refit:
            best_delay_point = delays_point[i_best]
            sigdriv_ = np.roll(sigdriv, best_delay_point, axis=1)
            sigdriv_imag_ = np.roll(sigdriv_imag, best_delay_point, axis=1)
            model.fit(sigin=sigin, sigdriv=sigdriv_,
                      sigdriv_imag=sigdriv_imag_, fs=self.fs,
                      train_weights=train_weights)
            self.best_model_ = model

        return self

    def plot(self, ax=None, write_tau=True):
        """
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure instance containing the plot.
        """
        check_is_fitted(self, 'neg_log_likelihood_')
        if ax is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            fig = ax.figure

        blue, green, red, purple, yellow, cyan = SEABORN_PALETTES['deep']

        i_best = np.nanargmin(self.neg_log_likelihood_)
        ax.plot(self.delays_ms_, self.neg_log_likelihood_, color=purple)
        ax.plot(self.delays_ms_[i_best], self.neg_log_likelihood_[i_best], 'D',
                color=red)
        ax.set_xlabel('Delay (ms)')
        ax.set_ylabel('Neg. log likelihood / T')
        ax.grid('on')

        if write_tau:
            ax.text(0.5, 0.80, r'$\mathrm{Estimated}$',
                    horizontalalignment='center', transform=ax.transAxes)
            ax.text(0.5, 0.66, r'$\tau_0 = %.0f \;\mathrm{ms}$' %
                    (self.delays_ms_[i_best], ), horizontalalignment='center',
                    transform=ax.transAxes)

        return fig
