import numpy as np

from .base_dar import BaseDAR


class HAR(BaseDAR):
    def last_model(self):
        return self._next_model(only_last=True)

    def next_model(self):
        return self._next_model(only_last=False)

    def _next_model(self, only_last=False):
        """Compute the AR model at successive orders

        Acts as a generator that stores the result in self.AR_
        Creates the models with orders from 0 to self.ordar

        Typical usage:
        for AR_ in A.next_model():
            A.AR_ = AR_
            A.ordar_ = AR_.shape[0]

        """
        if self.basis_ is None:
            raise ValueError('basis does not yet exist')

        # -------- get the training data
        sigin, basis, mask = self._get_train_data([self.sigin, self.basis_])
        selection = ~mask if mask is not None else None

        # -------- select signal, basis and regression signals
        n_epochs, n_points = sigin.shape
        m = self.n_basis
        K = self.ordar
        scale = 1.0 / n_points

        # -------- model at order 0
        AR_ = np.empty((0, m))
        yield AR_

        # -------- prepare regression signals
        sigreg = np.empty((K, n_epochs, n_points - K))
        for k in range(K):
            sigreg[k, :] = sigin[:, K - 1 - k:n_points - 1 - k]

        if selection is not None:
            sigreg *= selection[None, :, K:]
            masked_sigin = selection * sigin
        else:
            masked_sigin = sigin

        # -------- prepare auto/inter correlations
        R = scale * np.dot(sigreg.reshape(K, -1), sigreg.reshape(K, -1).T)
        r = scale * np.dot(masked_sigin[:, K:], sigreg.reshape(K, -1).T)

        # -------- loop on successive orders
        range_iter = [K - 1, ] if only_last else range(0, K)
        for k in range_iter:

            AR_ = np.zeros((k + 1, m))
            AR_[:, :1] = -np.linalg.solve(R[0:k + 1, 0:k + 1],
                                          r[:1, 0:k + 1].T)

            yield AR_

    def estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_

        """
        # compute the error on both the train and test part
        sigin = self.sigin
        n_epochs, n_points = sigin.shape

        residual = sigin.copy()
        for k in range(self.ordar_):
            residual[:, k + 1:n_points] += (self.AR_[k, 0] *
                                            sigin[:, 0:n_points - k - 1])
        self.residual_ = residual

    def develop(self, basis):
        """Compute the AR models and gains at instants fixed by newcols

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1+ordar, n_epochs, n_points)
        The size of Gcols is (1, n_epochs, n_points)

        """
        n_basis, n_epochs, n_points = basis.shape
        ordar_ = self.ordar_

        # -------- duplicate the AR part
        this_AR = np.empty(ordar_ + 1)
        this_AR[0] = 1.0
        if ordar_ > 0:
            this_AR[1:] = self.AR_[:, 0]
        AR_cols = this_AR[:, None, None] * np.ones((1, n_epochs, n_points))

        # -------- expand the gain on the basis
        G_cols = self.develop_gain(basis, squared=False, log=False)
        return (AR_cols, G_cols)

    def degrees_of_freedom(self):
        """overwrite since this is a special case"""
        return self.ordar_ + self.n_basis
