import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import model_selection

from .dar_model import DAR, extract_driver
from .utils.validation import check_is_fitted
from .utils.progress_bar import ProgressBar
from .utils.parallel import Parallel


class DARSklearn(DAR, BaseEstimator):
    """Different interface to DAR models, to use in scikit-learn's GridSearchCV

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
    >>> X = MultipleArray(sigin, sigdriv, sigdriv_imag)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)
    """

    def __init__(self, fs, max_ordar, ordar=1, ordriv=0, normalize=False,
                 ortho=True, center=True, iter_gain=10, eps_gain=1.0e-4,
                 progress_bar=False, use_driver_phase=False,
                 warn_gain_estimation_failure=False):
        super(DARSklearn, self).__init__(
            ordar=ordar, ordriv=ordriv, criterion=None, normalize=normalize,
            ortho=ortho, center=center, iter_gain=iter_gain, eps_gain=eps_gain,
            progress_bar=progress_bar, use_driver_phase=use_driver_phase,
            max_ordar=max_ordar,
            warn_gain_estimation_failure=warn_gain_estimation_failure)

        self.fs = fs

    def fit(self, X, y=None):
        """Fit the DAR model

        Parameters
        ----------
        X : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.

        y : None
            Not used

        Returns
        -------
        self
        """
        sigin, sigdriv, sigdriv_imag = X.to_list()
        super(DARSklearn, self).fit(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                    sigdriv_imag=sigdriv_imag)
        return self

    def fit_transform(self, X, y=None):
        """Fit the model and transform sigin into the residuals

        Parameters
        ----------
        X : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.

        y : None
            Not used

        Returns
        -------
        residuals : array, shape (n_epochs, n_points)
            Residuals of sigin unexplained by the model
        """
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Transform sigin into the residuals unexplained by the fitted model

        Parameters
        ----------
        X : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.

        y : None
            Not used

        Returns
        -------
        residuals : array, shape (n_epochs, n_points)
            Residuals of sigin unexplained by the fitted model.
        """
        check_is_fitted(self, 'AR_')
        sigin, sigdriv, sigdriv_imag = X.to_list()
        return super(DARSklearn,
                     self).transform(sigin=sigin, sigdriv=sigdriv, fs=self.fs,
                                     sigdriv_imag=sigdriv_imag)

    def score(self, X, y=None):
        """Difference in log-likelihood of this model and a model AR(0)

        We subtract the log-likelihood of an autoregressive model at order 0,
        in order to have a reference stable over cross-validation splits.

        Parameters
        ----------
        X : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.

        y : None
            Not used

        Returns
        -------
        score : float
            Difference in log-likelihood of this model and a model AR(0).
            The high is the better.
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
    >>> from pactools.grid_search import MultipleArray
    >>> param_grid = {
    ...     'dar__ordar': [10, 20, 30],
    ...     'dar__ordriv': [0, 1, 2],
    ...     'add__delay': [-10, 0, 10]
    ... }
    >>> model = Pipeline(steps=[
    ...     ('add', AddDriverDelay()),
    ...     ('dar', DARSklearn(fs=fs, max_ordar=30)),
    ... ])
    >>> X = MultipleArray(sigin, sigdriv, sigdriv_imag)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)
    """

    def __init__(self, delay=0, n_decay=30):
        self.delay = delay
        self.n_decay = n_decay

    def fit(self, X, y=None):
        """No fit is needed"""
        return self

    def transform(self, X, y=None):
        """Apply the temporal delay on sigdriv and sigdriv_imag

        Parameters
        ----------
        X : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.

        y : None
            Not used

        Returns
        -------
        Xt : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.
        """
        sigin, sigdriv, sigdriv_imag = X.to_list()

        # window decay for temporal continuity after np.roll
        n_decay = self.n_decay
        window = np.blackman(n_decay * 2 - 1)[:n_decay]

        def window_and_roll(sig):
            sig = sig.copy()  # copy to avoid modifying original
            sig[..., :n_decay] *= window
            sig[..., -n_decay:] *= window[::-1]
            sig = np.roll(sig, self.delay, axis=-1)
            return sig

        sigdriv = window_and_roll(sigdriv)
        if sigdriv_imag is not None:
            sigdriv_imag = window_and_roll(sigdriv_imag)

        return MultipleArray(sigin, sigdriv, sigdriv_imag)


class ExtractDriver(BaseEstimator, TransformerMixin):
    """Returns bandpass filtered and a highpass filtered signals

    Parameters
    ----------
    fs : float
        Sampling frequency

    low_fq : float
        Center frequency of bandpass filters.

    max_low_fq : float
        Maximum `low_fq` over a potential cross-validation scheme.

    low_fq_width : float
        Bandwidth of the bandpass filters.

    fill : in {0, 1, 2}
        Filling strategy for the modeled signal `sigin`:
        0 : keep the signal unchanged: sigin = high_sig
        1 : remove the bandpass filtered signal: sigin = high_sig - sigdriv
        2 : remove and replace by bandpass filtered Gaussian white noise

    whitening : in {'before', 'after', None}
        Define when the whitening is done compared to the filtering.

    ordar : int >= 0
        Order of the AR model used for whitening

    normalize : boolean
        Whether to scale the signals to have unit norm `sigin`.
        Both drivers `sigdriv` and `sigdriv_imag` are scaled with the same
        scales.

    extract_complex : boolean
        Whether to extract a complex driver (`sigdriv` and `sigdriv_imag`)

    random_state : None, int or np.random.RandomState instance
        Seed or random number generator for the white noise filling strategy.

    enf : float
        Electric Network Frequency

    Examples
    --------
    >>> from sklearn.model_selection import GridSearchCV
    >>> from sklearn.pipeline import pipeline
    >>> from pactools.grid_search import DARSklearn, ExtractDriver
    >>> from pactools.grid_search import MultipleArray
    >>> param_grid = {
    ...     'dar__ordar': [10, 20, 30],
    ...     'dar__ordriv': [0, 1, 2],
    ...     'driver__low_fq': [4.0, 4.5, 5.0]
    ... }
    >>> model = Pipeline(steps=[
    ...     ('driver', ExtractDriver(fs=fs, max_low_fq=5.0)),
    ...     ('dar', DARSklearn(fs=fs, max_ordar=30)),
    ... ])
    >>> X = MultipleArray(low_sig, None)
    >>> gscv.fit(X)
    >>> print(gscv.cv_results_)

    """

    def __init__(self, fs, low_fq, max_low_fq, low_fq_width=1.0, fill=2,
                 whitening='after', ordar=10, normalize=False,
                 extract_complex=True, random_state=None, enf=50.):
        self.fs = fs
        self.low_fq = low_fq
        self.max_low_fq = max_low_fq
        self.low_fq_width = low_fq_width
        self.fill = fill
        self.whitening = whitening
        self.ordar = ordar
        self.normalize = normalize
        self.extract_complex = extract_complex
        self.random_state = random_state
        self.enf = enf

    def fit(self, X, y=None):
        """No fit is needed"""
        return self

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : MultipleArray with two signals:
            low_sig : array, shape (n_epochs, n_points)
                Input data for the phase signal

            high_sig : array or None, shape (n_epochs, n_points)
                Input data for the amplitude signal.
                If None, we use low_sig for both signals.

        y : None
            Not used

        Returns
        -------
        Xt : MultipleArray with three signals:
            sigin : array, shape (n_epochs, n_points)
                Signal that is to be modeled

            sigdriv : array, shape (n_epochs, n_points)
                Signal that drives the model

            sigdriv_imag : array, shape (n_epochs, n_points), or None
                Second driver, containing the imaginary part of the driver
                Is not None only if extract_complex is True.
        """
        low_sig, high_sig = X.to_list()

        # handle 3D and more arrays, such as (n_epochs, n_channels, n_points)
        shape = low_sig.shape
        if len(shape) > 2:
            low_sig = low_sig.reshape(-1, shape[-1])
            if high_sig is not None:
                high_sig.reshape(-1, shape[-1])

        n_epochs = low_sig.shape[0]
        if high_sig is None:
            sigs = low_sig
        else:
            # hack to call only once extract
            high_sig = np.atleast_2d(high_sig)
            sigs = np.r_[low_sig, high_sig]

        filtered_signals = extract_driver(
            sigs=sigs, fs=self.fs, low_fq=self.low_fq, n_cycles=None,
            bandwidth=self.low_fq_width, fill=self.fill,
            whitening=self.whitening, ordar=self.ordar,
            normalize=self.normalize, extract_complex=self.extract_complex,
            random_state=self.random_state, max_low_fq=self.max_low_fq,
            enf=self.enf)

        if self.extract_complex:
            filtered_low, filtered_high, filtered_low_imag = filtered_signals
        else:
            filtered_low, filtered_high = filtered_signals

        sigdriv_imag = None
        if high_sig is None:
            sigin = np.array(filtered_high)
            sigdriv = np.array(filtered_low)
            if self.extract_complex:
                sigdriv_imag = np.array(filtered_low_imag)
        else:
            sigin = np.array(filtered_high[n_epochs:])
            sigdriv = np.array(filtered_low[:n_epochs])
            if self.extract_complex:
                sigdriv_imag = np.array(filtered_low_imag[:n_epochs])

        # recover initial shape
        sigin = sigin.reshape(shape)
        sigdriv = sigdriv.reshape(shape)
        if sigdriv_imag is not None:
            sigdriv_imag = sigdriv_imag.reshape(shape)

        return MultipleArray(sigin, sigdriv, sigdriv_imag)


class MultipleArray(object):
    """Store a list of arrays into a single object and handle indexing for them

    This is to be used as X in GridSearchCV(DARSklearn()).fit(X)
    Some of the arrays can be None.

    Parameters
    ----------
    *signals : several ndarray or None
        Stored signals

    Examples
    --------
    >>> import numpy as np
    >>> from pactools.grid_search import MultipleArray
    >>> ma = MultipleArray(np.ones(3, 10), None, None, np.zeros(3, 10, 2))
    >>> print(ma)
    >>> print(ma[1:])

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


class GridSearchCVProgressBar(model_selection.GridSearchCV):
    """Monkey patch Parallel to have a progress bar during grid search"""

    def _get_param_iterator(self):
        """Return ParameterGrid instance for the given param_grid"""

        iterator = super(GridSearchCVProgressBar, self)._get_param_iterator()
        iterator = list(iterator)
        n_candidates = len(iterator)

        cv = model_selection._split.check_cv(self.cv, None)
        n_splits = getattr(cv, 'n_splits', 3)
        max_value = n_candidates * n_splits

        class ParallelProgressBar(Parallel):
            def __call__(self, iterable):
                bar = ProgressBar(max_value=max_value, title='GridSearchCV')
                iterable = bar(iterable)
                return super(ParallelProgressBar, self).__call__(iterable)

        # Monkey patch
        model_selection._search.Parallel = ParallelProgressBar

        return iterator
