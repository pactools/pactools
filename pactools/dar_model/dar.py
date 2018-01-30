import numpy as np

from .base_dar import BaseDAR


class DAR(BaseDAR):
    """
    A driven auto-regressive (DAR) model, as described in [1].

    This model uses the simple parametrization:

    .. math:: y(t) + \\sum_{i=1}^p a_i(t) y(t-i)= \\sigma(t)\\varepsilon(t)

    with:

    .. math:: a_i(t) = \\sum_{k=0}^m a_{ik} x(t)^k

    and:

    .. math:: \\log{\\sigma(t)} = \\sum_{k=0}^m b_{k} x(t)^k

    References
    ----------
    [1] Dupre la Tour, T. , Tallot, L., Grabot, L., Doyere, V., van Wassenhove,
    V., Grenier, Y., & Gramfort, A. (2017). Non-linear Auto-Regressive Models
    for Cross-Frequency Coupling in Neural Time Series. bioRxiv, 159731.

    Parameters
    ----------
    ordar : int >= 0
        Order of the autoregressive model (p)

    ordriv : int >= 0
        Order of the taylor expansion for sigdriv (m)

    criterion : None or string in ('bic', 'aic', 'logl')
        If not None, select the criterion used for model selection.

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
    """

    def _last_model(self):
        return self._estimate_model(only_last=True)

    def _next_model(self):
        return self._estimate_model(only_last=False)

    def _estimate_model(self, only_last=False):
        """Compute the AR model at successive orders

        Acts as a generator that returns AR_
        Creates the models with orders from 0 to self.ordar

        Example
        -------
        for AR_ in A._estimate_model():
            A.AR_ = AR_
            A.ordar_ = AR_.shape[0]
        """
        if self.basis_ is None:
            raise ValueError('basis does not yet exist')

        # -------- get the training data
        sigin, basis, weights = self._get_train_data([self.sigin, self.basis_])

        # mask the signal
        if weights is not None:
            weights = np.sqrt(weights)
            w_basis = weights * basis
            w_sigin = weights * sigin
        else:
            w_basis = basis
            w_sigin = sigin

        # -------- select signal, basis and regression signals
        n_epochs, n_points = sigin.shape
        m = self.n_basis
        K = self.ordar

        # -------- model at order 0
        if not only_last or K == 0:
            AR_ = np.empty((0, m))
            yield AR_

        if K > 0:
            # -------- prepare regression signals
            sigreg = np.empty((K * m, n_epochs, n_points - K))
            for k in range(K):
                sigreg[k * m:(k + 1) * m, :, :] = (
                    w_basis[:, :, K:] * sigin[:, K - 1 - k:n_points - 1 - k])

            # -------- prepare auto/inter correlations
            scale = 1.0 / n_points
            R = scale * np.dot(
                sigreg.reshape(K * m, -1), sigreg.reshape(K * m, -1).T)
            r = scale * np.dot(w_sigin[:, K:].reshape(1, -1),
                               sigreg.reshape(K * m, -1).T)

            # -------- loop on successive orders
            range_iter = [K - 1, ] if only_last else range(K)
            for k in range_iter:
                km = k * m
                km_m = km + m

                AR_ = -np.linalg.solve(R[0:km_m, 0:km_m], r[:1, 0:km_m].T)

                AR_ = np.reshape(AR_, (k + 1, m))
                yield AR_

    def _estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and self.AR_
        """
        # compute the error on both the train and test part
        sigin = self.sigin
        basis = self.basis_

        n_epochs, n_points = sigin.shape

        prediction = np.zeros((self.n_basis, n_epochs, n_points))
        for k in range(self.ordar_):
            prediction[:, :, k + 1:n_points] += (
                self.AR_[k, :][:, None, None] * sigin[:, 0:n_points - k - 1])

        residual = sigin.copy()
        residual += np.einsum('i...,i...', prediction, basis)
        self.residual_ = residual

    def _develop(self, basis):
        """Compute the AR models and gains at instants fixed by newcols

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1 + ordar_, n_epochs, n_points)
        The size of Gcols is (1, n_epochs, n_points)

        """
        n_basis, n_epochs, n_points = basis.shape
        ordar_ = self.ordar_

        # -------- expand on the basis
        AR_cols_ones = np.ones((1, n_epochs, n_points))
        if ordar_ > 0:
            AR_cols = np.tensordot(self.AR_, basis, axes=([1], [0]))
            AR_cols = np.vstack((AR_cols_ones, AR_cols))
        else:
            AR_cols = AR_cols_ones
        G_cols = self._develop_gain(basis, squared=False, log=False)
        return (AR_cols, G_cols)


class AR(DAR):
    """
    A simple linear auto-regressive (AR) model:

    .. math:: y(t) + \\sum_{i=1}^p a_i y(t-i)= \\varepsilon(t)
    """

    def __init__(self, ordar=1, ordriv=0, *args, **kwargs):
        super(AR, self).__init__(ordar=ordar, ordriv=0, *args, **kwargs)
