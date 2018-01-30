import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .dar_model import DAR
from .utils.validation import check_is_fitted


class DARSklearn(DAR, BaseEstimator):
    """Different interface to DAR models, to use scikit-learn's GridSearchCV

    Parameters
    ----------
    fs : float
        Sampling frequency

    max_ordar : int >= 0
        Maximum ordar over a potential cross-validation scheme.
        The log-likelihood does not use the first max_ordar points in its
        computation, to fairly compare different ordar over cross-validation.

    ordar : int >= 0
        Order of the autoregressive model (p)

    ordriv : int >= 0
        Order of the taylor expansion for sigdriv (m)

    normalize : boolean
        If True, the basis vectors are normalized to unit energy.

    ortho : boolean
        If True, the basis vectors are orthogonalized.

    center : boolean
        If True, we subtract the mean in sigin

    iter_gain : int >=0
        Maximum number of iteration in gain estimation

    eps_gain : float >= 0
        Threshold to stop iterations in gain estimation

    use_driver_phase : boolean
        If True, we divide the driver by its instantaneous amplitude.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from pactools.grid_search import DARSklearn
    >>> model = DARSklearn(fs=fs)
    >>> param_grid = {'ordar': [10, 20, 30], 'ordriv': [0, 1, 2]}
    >>> gscv = GridSearchCV(model, param_grid=param_grid)
    >>> X = MultipleArray(high_sig, low_sig, low_sig_imag)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)
    """
    def __init__(self, fs, max_ordar, ordar=1, ordriv=0,
                 normalize=False, ortho=True, center=True, iter_gain=10,
                 eps_gain=1.0e-4, progress_bar=False, use_driver_phase=False,
                 warn_gain_estimation_failure=False):
        super(DARSklearn, self).__init__(
            ordar=ordar, ordriv=ordriv, criterion=None,
            normalize=normalize, ortho=ortho, center=center,
            iter_gain=iter_gain, eps_gain=eps_gain, progress_bar=progress_bar,
            use_driver_phase=use_driver_phase, max_ordar=max_ordar,
            warn_gain_estimation_failure=warn_gain_estimation_failure)

        self.fs = fs

    def fit(self, X, y=None):
        sigin, sigdriv, sigdriv_imag = X.to_list()
        super(DARSklearn, self).fit(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                    sigdriv_imag=sigdriv_imag)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        check_is_fitted(self, 'AR_')
        sigin, sigdriv, sigdriv_imag = X.to_list()
        return super(DARSklearn,
                     self).transform(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                     sigdriv_imag=sigdriv_imag)

    def score(self, X, y=None):
        """Difference in log-likelihood of this model and a model AR(0)

        We subtract the log-likelihood of an autoregressive model at order 0,
        in order to have a reference stable over cross-validation splits.
        """
        self.transform(X, y)
        score = self.logl * self.tmax

        max_ordar = self.max_ordar
        if max_ordar is None:
            raise ValueError(
                'max_ordar should not be zero, since it biases the grid search'
                ' over ordar. Set max_ordar to the maximum value of ordar in '
                'the grid search, or bigger.')
        # remove the log likelihood of an AR(0) to better compare likelihoods
        score -= self._estimate_log_likelihood_ref(skip=max_ordar)[0]
        return score


class AddDriverDelay(BaseEstimator, TransformerMixin):
    """Transformer which adds a time delay in the driver of a DAR model

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import pipeline
    >>> from pactools.grid_search import DARSklearn, AddDriverDelay
    >>> param_grid = {
    ...     'dar__ordar': [10, 20, 30],
    ...     'dar__ordriv': [0, 1, 2],
    ...     'add__delay': [-10, 0, 10]
    ... }
    >>> model = Pipeline(steps=[
    ...     ('add', AddDriverDelay()),
    ...     ('dar', DARSklearn(fs=fs, max_ordar=30)),
    ... ])
    >>> X = MultipleArray(high_sig, low_sig, low_sig_imag)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)
    """
    def __init__(self, delay=0, n_decay=30):
        self.delay = delay
        self.n_decay = n_decay

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        sigin, sigdriv, sigdriv_imag = X.to_list()

        # window decay of sigdriv for continuity after np.roll
        n_decay = self.n_decay
        window = np.blackman(n_decay * 2 - 1)[:n_decay]

        sigdriv = sigdriv.copy()  # copy to avoid modifying original
        sigdriv[:, :n_decay] *= window
        sigdriv[:, -n_decay:] *= window[::-1]
        sigdriv = np.roll(sigdriv, self.delay, axis=1)

        sigdriv_imag = sigdriv_imag.copy()  # copy to avoid modifying original
        sigdriv_imag[:, :n_decay] *= window
        sigdriv_imag[:, -n_decay:] *= window[::-1]
        sigdriv_imag = np.roll(sigdriv_imag, self.delay, axis=1)

        return MultipleArray(sigin, sigdriv, sigdriv_imag)


class MultipleArray(object):
    """Store a list of arrays into a single object and handle indexing for them

    This is to be used as X in GridSearchCV(DARSklearn()).fit(X)
    Some of the arrays can be None.

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from pactools.grid_search import DARSklearn
    >>> model = DARSklearn(fs=fs)
    >>> param_grid = {'ordar': [10, 20, 30], 'ordriv': [0, 1, 2]}
    >>> gscv = GridSearchCV(model, param_grid=param_grid)
    >>> X = MultipleArray(sigin, sigdriv, sigdriv_imag)
    >>> gscv.fit(X)
    """

    def __init__(self, *signals):
        self.signals = signals

    @property
    def shape(self):
        """Only return the shape of the first array"""
        return self.signals[0].shape

    def take(self, *args, **kwargs):
        tmp = [
            X.take(*args, **kwargs) if X is not None else None
            for X in self.signals
        ]
        return MultipleArray(*tmp)

    def __str__(self):
        return 'MultipleArray:\n' + '\n'.join([str(X) for X in self.signals])

    def __repr__(self):
        return 'MultipleArray:\n' + '\n'.join([repr(X) for X in self.signals])

    def __getitem__(self, indices):
        tmp = [X[indices] if X is not None else None for X in self.signals]
        return MultipleArray(*tmp)

    def to_list(self):
        return self.signals
