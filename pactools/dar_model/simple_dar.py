from __future__ import print_function

import numpy as np

from .baseAR import BaseAR


class SimpleDAR(BaseAR):

    def last_model(self):
        return self._next_model(only_last=True)

    def next_model(self):
        return self._next_model(only_last=False)

    def _next_model(self, only_last=False):
        """Compute the AR model at successive orders

        Acts as a generator that returns AR_
        Creates the models with orders from 0 to self.ordar

        Example
        -------
        for AR_ in A.next_model():
            A.AR_ = AR_
            A.ordar_ = AR_.shape[0]
        """
        if self.basis_ is None:
            raise ValueError('drivenAR: basis does not yet exist')

        # -------- get the training data
        mask, sigin = self.get_train_data(self.sigin)
        _, basis = self.get_train_data(self.basis_)

        # mask the signal (same behavior as sample weights)
        if self.mask is not None:
            masked_basis = mask * basis
            masked_sigin = mask * sigin
        else:
            masked_basis = basis
            masked_sigin = sigin

        # -------- select signal, basis and regression signals
        n_epochs, n_points = sigin.shape
        m = self.n_basis
        K = self.ordar

        # -------- model at order 0
        AR_ = np.empty((0, m))
        yield AR_

        # -------- prepare regression signals
        sigreg = np.empty((K * m, n_epochs, n_points - K))
        for k in range(K):
            sigreg[k * m:(k + 1) * m, :, :] = (
                masked_basis[:, :, K:] * sigin[:, K - 1 - k:n_points - 1 - k])

        # -------- prepare auto/inter correlations
        scale = 1.0 / n_points
        R = scale * np.dot(sigreg.reshape(K * m, -1),
                           sigreg.reshape(K * m, -1).T)
        r = scale * np.dot(masked_sigin[:, K:].reshape(1, -1),
                           sigreg.reshape(K * m, -1).T)

        # -------- loop on successive orders
        range_iter = [K - 1, ] if only_last else range(K)
        for k in range_iter:
            km = k * m
            km_m = km + m

            AR_ = -np.linalg.solve(R[0:km_m, 0:km_m], r[:1, 0:km_m].T)

            AR_ = np.reshape(AR_, (k + 1, m))
            yield AR_

    def estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and self.AR_
        """
        # -------- get the left-out data
        _, sigin = self.get_test_data(self.sigin)
        _, basis = self.get_test_data(self.basis_)
        n_epochs, n_points = sigin.shape

        prediction = np.zeros((self.n_basis, n_epochs, n_points))
        for k in range(self.ordar_):
            prediction[:, :, k + 1:n_points] += (
                self.AR_[k, :][:, None, None] * sigin[:, 0:n_points - k - 1])

        residual = sigin.copy()
        residual += np.einsum('i...,i...', prediction, basis)
        self.residual_ = residual

    def develop(self, basis, sigdriv):
        """Compute the AR models and gains at instants fixed by newcols

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1 + ordar, n_epochs, n_points)
        The size of Gcols is (1, n_epochs, n_points)

        """
        n_basis, n_epochs, n_points = basis.shape
        ordar = self.get_ordar()

        # -------- expand on the basis
        AR_cols_ones = np.ones((1, n_epochs, n_points))
        if ordar > 0:
            AR_cols = np.tensordot(self.AR_, basis, axes=([1], [0]))
            AR_cols = np.vstack((AR_cols_ones, AR_cols))
        else:
            AR_cols = AR_cols_ones
        G_cols = self.develop_gain(basis, squared=False, log=False)
        return (AR_cols, G_cols)
