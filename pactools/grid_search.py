import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .dar_model import DAR
from .utils.validation import check_is_fitted


class DARSklearn(DAR, BaseEstimator):
    """Different interface to DAR models, to use scikit-learn's GridSearchCV

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from pactools.grid_search import DARSklearn
    >>>
    >>> model = DARSklearn(fs=fs)
    >>> param_grid = {ordar:  [10, 20, 30], ordriv: [0, 1, 2]}
    >>> gscv = GridSearchCV(model, param_grid=param_grid)
    >>> X = MultipleArray(high_sig, low_sig, low_sig_imag)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)

    """
    def __init__(self, fs, use_imag=True, ordar=1, ordriv=0, criterion=None,
                 normalize=False, ortho=True, center=True, iter_gain=10,
                 eps_gain=1.0e-4, progress_bar=False, use_driver_phase=False,
                 max_ordar=None, warn_gain_estimation_failure=False):
        super(DARSklearn, self).__init__(
            ordar=ordar, ordriv=ordriv, criterion=criterion,
            normalize=normalize, ortho=ortho, center=center,
            iter_gain=iter_gain, eps_gain=eps_gain, progress_bar=progress_bar,
            use_driver_phase=use_driver_phase, max_ordar=max_ordar,
            warn_gain_estimation_failure=warn_gain_estimation_failure)

        self.fs = fs
        self.use_imag = use_imag

    def fit(self, X, y=None):
        sigin, sigdriv, sigdriv_imag = X.to_list()
        sigdriv_imag = sigdriv_imag if self.use_imag else None
        super(DARSklearn, self).fit(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                    sigdriv_imag=sigdriv_imag)
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        check_is_fitted(self, 'AR_')
        sigin, sigdriv, sigdriv_imag = X.to_list()
        sigdriv_imag = sigdriv_imag if self.use_imag else None
        return super(DARSklearn,
                     self).transform(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                     sigdriv_imag=sigdriv_imag)

    def score(self, X, y=None):
        self.transform(X, y)
        return self.logl * self.tmax


class AddDriverDelay(BaseEstimator, TransformerMixin):
    """Transformer which adds a time delay in the driver of a DAR model

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import pipeline
    >>> from pactools.grid_search import DARSklearn, AddDriverDelay
    >>>
    >>> model = Pipeline(steps=[('add', AddDriverDelay()),
    ...                         ('dar', DARsklearn(fs=fs))])
    >>> param_grid = {'dar__ordar': [10, 20, 30],
    ...               'dar__ordriv': [0, 1, 2],
    ...               'add__delay': [-10, 0, 10]}
    >>> gscv = GridSearchCV(model, param_grid=param_grid)
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
    >>>
    >>> model = DARSklearn(fs=fs)
    >>> param_grid = {ordar:  [10, 20, 30], ordriv: [0, 1, 2]}
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
