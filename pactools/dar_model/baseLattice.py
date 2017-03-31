from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg

from .base_dar import BaseDAR
from ..utils.arma import ki2ai


class BaseLattice(BaseDAR):
    __metaclass__ = ABCMeta

    def __init__(self, iter_newton=0, eps_newton=0.001, **kwargs):
        """Creates a base Lattice model with Taylor expansion

        iter_newton : maximum number of Newton-Raphson iterations
        eps_newton  : threshold to stop Newton-Raphson iterations
        """
        super(BaseLattice, self).__init__(**kwargs)
        self.iter_newton = iter_newton
        self.eps_newton = eps_newton

    def synthesis(self, sigdriv=None, sigin_init=None):
        """Create a signal from fitted model

        sigdriv      : driving signal
        sigdriv_init : shape (1, self.ordar_) previous point of the past

        """
        ordar_ = self.ordar_
        if sigdriv is None:
            sigdriv = self.sigdriv
            try:
                newbasis = self.basis_
            except:
                newbasis = self.make_basis()
        else:
            newbasis = self.make_basis(sigdriv=sigdriv)

        n_epochs, n_points = sigdriv.shape

        if sigin_init is None:
            sigin_init = np.random.randn(1, ordar_)
        sigin_init = np.atleast_2d(sigin_init)
        if sigin_init.shape[1] < ordar_:
            raise (ValueError, "initialization of incorrect length: %d instead"
                   "of %d (self.ordar_)" % (sigin_init.shape[1], ordar_))

        # initialization
        signal = np.zeros(n_points)
        signal[:ordar_] = sigin_init[0, -ordar_:]

        # compute basis corresponding to sigdriv
        newbasis = self.make_basis(sigdriv=sigdriv)

        # synthesis of the signal
        random_sig = np.random.randn(n_epochs, n_points - ordar_)
        burn_in = np.random.randn(n_epochs, ordar_)
        sigout = self.synthesize(random_sig, burn_in=burn_in,
                                 newbasis=newbasis)

        return sigout

    # ------------------------------------------------ #
    # Functions that are added to the derived class    #
    # ------------------------------------------------ #
    def synthesize(self, random_sig, burn_in=None, newbasis=None):
        """Apply the inverse lattice filter to synthesize
        an AR signal

        random_sig : array containing the excitation starting at t=0
        burn_in    : auxiliary excitations before t=0
        newbasis   : if None, we use the basis used for fitting (self.basis_)
                    else, we use the given basis.

        returns
        -------
        sigout : array containing the synthesized signal

        """
        if newbasis is None:
            basis = self.basis_
        else:
            basis = newbasis
        # -------- select ordar (prefered: ordar_)
        ordar = self.ordar_

        # -------- prepare excitation signal and burn in phase
        if burn_in is None:
            tburn = 0
            excitation = random_sig
        else:
            tburn = burn_in.size
            excitation = np.hstack((burn_in, random_sig))

        n_basis, n_epochs, n_points = basis.shape
        n_epochs, n_points_minus_ordar_ = random_sig.shape

        # -------- allocate arrays for forward and backward residuals
        e_forward = np.zeros(n_epochs, ordar + 1)
        e_backward = np.zeros(n_epochs, ordar + 1)

        # -------- prepare parcor coefficients
        parcor_list = self.develop_parcor(self.AR_, basis)
        parcor_list = self.decode(parcor_list)

        gain = self.develop_gain(basis, squared=False, log=False)

        # -------- create the output signal
        sigout = np.zeros(n_epochs, n_points_minus_ordar_)
        for t in range(-tburn, n_points_minus_ordar_):
            e_forward[:, ordar] = excitation[:, t + tburn] * gain[0, max(t, 0)]
            for p in range(ordar, 0, -1):
                e_forward[:, p - 1] = (
                    e_forward[:, p] - parcor_list[p - 1, :, max(t, 0)] *
                    e_backward[:, p - 1])
            for p in range(ordar, 0, -1):
                e_backward[:, p] = (
                    e_backward[:, p - 1] + parcor_list[p - 1, :, max(t, 0)] *
                    e_forward[:, p - 1])
            e_backward[:, 0] = e_forward[:, 0]
            sigout[:, max(t, 0)] = e_forward[:, 0]

        return sigout

    def cell(self, parcor_list, forward_residual, backward_residual):
        """Apply a single cell of the direct lattice filter
        to whiten the original signal

        parcor_list       : array, lattice coefficients
        forward_residual  : array, forward residual from previous cell
        backward_residual : array, backward residual from previous cell

        the input arrays should have same shape as self.sigin

        returns:
        forward_residual  : forward residual after this cell
        backward_residual : backward residual after this cell

        """
        f_residual = np.copy(forward_residual)
        b_residual = np.zeros(backward_residual.shape)
        delayed = np.copy(backward_residual[:, 0:-1])

        # -------- apply the cell
        b_residual[:, 1:] = delayed + (parcor_list[:, 1:] * f_residual[:, 1:])
        b_residual[:, 0] = parcor_list[:, 0] * f_residual[:, 0]
        f_residual[:, 1:] += parcor_list[:, 1:] * delayed
        return f_residual, b_residual

    def whiten(self):
        """Apply the direct lattice filter to whiten the original signal

        returns:
        residual : array containing the residual (whitened) signal
        backward : array containing the backward residual (whitened) signal

        """
        # compute the error on both the train and test part
        sigin = self.sigin
        basis = self.basis_

        # -------- select ordar (prefered: ordar_)
        ordar = self.ordar_
        n_epochs, n_points = sigin.shape

        residual = np.copy(sigin)
        backward = np.copy(sigin)
        for k in range(ordar):
            parcor_list = self.develop_parcor(self.AR_[k], basis)
            parcor_list = self.decode(parcor_list)
            residual, backward = self.cell(parcor_list, residual, backward)

        return residual, backward

    def develop_parcor(self, LAR, basis):
        single_dim = LAR.ndim == 1
        LAR = np.atleast_2d(LAR)
        # n_basis, n_epochs, n_points = basis.shape
        # ordar, n_basis = LAR.shape
        # ordar, n_epochs, n_points = lar_list.shape
        lar_list = np.tensordot(LAR, basis, axes=([1], [0]))

        if single_dim:
            # n_epochs, n_points = lar_list.shape
            lar_list = lar_list[0, :, :]
        return lar_list

    # ------------------------------------------------ #
    # Functions that should be overloaded              #
    # ------------------------------------------------ #
    @abstractmethod
    def decode(self, lar):
        """Extracts parcor coefficients from encoded version (e.g. LAR)

        lar : array containing the encoded coefficients

        returns:
        ki : array containing the decoded coefficients (same size as lar)

        """
        pass

    @abstractmethod
    def encode(self, ki):
        """Encodes parcor coefficients to parameterized coefficients
        (for instance LAR)

        ki : array containing the original parcor coefficients

        returns:
        lar : array containing the encoded coefficients (same size as ki)

        """
        pass

    @abstractmethod
    def common_gradient(self, p, ki):
        """Compute common factor in gradient. The gradient is computed as
        G[p] = sum from t=1 to T {g[p,t] * F(t)}
        where F(t) is the vector of driving signal and its powers
        g[p,t] = (e_forward[p, t] * e_backward[p-1, t-1]
                  + e_backward[p, t] * e_forward[p-1, t]) * phi'[k[p,t]]
        phi is the encoding function, and phi' is its derivative.

        p  : order corresponding to the current lattice cell
        ki : array containing the original parcor coefficients

        returns:
        g : array containing the factors (size (n_epochs, n_points - 1))

        """
        pass

    @abstractmethod
    def common_hessian(self, p, ki):
        """Compute common factor in Hessian. The Hessian is computed as
        H[p] = sum from t=1 to T {F(t) * h[p,t] * F(t).T}
        where F(t) is the vector of driving signal and its powers
        h[p,t] = (e_forward[p, t-1]**2 + e_backward[p-1, t-1]**2)
                    * phi'[k[p,t]]**2
                 + (e_forward[p, t] * e_backward[p-1, t-1]
                    e_backward[p, t] * e_forward[p-1, t]) * phi''[k[p,t]]
        phi is the encoding function, phi' is its first derivative,
        and phi'' is its second derivative.

        p  : order corresponding to the current lattice cell
        ki : array containing the original parcor coefficients

        returns:
        h : array containing the factors (size (n_epochs, n_points - 1))

        """
        pass

    # ------------------------------------------------ #
    # Functions that overload abstract methods         #
    # ------------------------------------------------ #
    def next_model(self):
        """Compute the AR model at successive orders

        Acts as a generator that stores the result in self.AR_
        Creates the models with orders from 0 to self.ordar

        Typical usage:
        for AR_ in A.next_model():
            A.AR_ = AR_
            A.ordar_ = AR_.shape[0]

        """
        if self.basis_ is None:
            raise ValueError('%s: basis_ does not yet exist' %
                             self.__class__.__name__)

        # --------  get the training data
        sigin, basis, mask = self._get_train_data([self.sigin, self.basis_])

        selection = ~mask if mask is not None else None
        if selection is not None:
            masked_basis = basis * selection
        else:
            masked_basis = basis

        # -------- select signal, basis and regression signals
        n_basis, n_epochs, n_points = basis.shape
        ordar_ = self.ordar
        scale = 1.0 / n_points

        # -------- prepare residual signals
        forward_res = np.copy(sigin)
        backward_res = np.copy(sigin)

        self.forward_residual = np.empty((ordar_ + 1, n_epochs, n_points))
        self.backward_residual = np.empty((ordar_ + 1, n_epochs, n_points))
        self.forward_residual[0] = forward_res
        self.backward_residual[0] = backward_res

        # -------- model at order 0
        AR_ = np.empty((0, self.n_basis))
        yield AR_

        # -------- loop on successive orders
        for k in range(0, ordar_):
            # -------- prepare initial estimation (driven parcor)
            if selection is not None:
                # the mask and basis are not delay as backward_res /!\
                forward_res = selection[:, k + 1:] * forward_res[:, k + 1:]
                backward_res = selection[:, k + 1:] * backward_res[:, k:-1]
            else:
                forward_res = forward_res[:, k + 1:]
                backward_res = backward_res[:, k:-1]

            forward_regressor = basis[:, :, k + 1:] * forward_res
            backward_regressor = basis[:, :, k + 1:] * backward_res

            # this reshape method will throw an error if a copy is needed
            forward_regressor.shape = (n_basis, -1)
            backward_regressor.shape = (n_basis, -1)
            backward_res.shape = (1, -1)

            R = (np.dot(forward_regressor, forward_regressor.T) + np.dot(
                backward_regressor, backward_regressor.T))
            r = np.dot(forward_regressor, backward_res.T)
            backward_res.shape = (n_epochs, -1)

            R *= scale
            r *= scale * 2.0
            parcor = -np.linalg.solve(R, r).T

            # n_basis, n_epochs, n_points = basis.shape
            # n_epochs, n_points = parcor_list.shape
            parcor_list = self.develop_parcor(parcor.ravel(), basis)
            parcor_list = np.maximum(parcor_list, -0.999999)
            parcor_list = np.minimum(parcor_list, 0.999999)
            if selection is not None:
                parcor_list *= selection

            R = scale * np.dot(masked_basis[:, :, k:].reshape(n_basis, -1),
                               masked_basis[:, :, k:].reshape(n_basis, -1).T)
            r = scale * np.dot(masked_basis[:, :, k:].reshape(n_basis, -1),
                               self.encode(parcor_list[:, k:]).reshape(-1, 1))
            LAR = np.linalg.solve(R, r).T

            # -------- Newton-Raphson refinement
            dLAR = np.inf
            itnum = 0
            while (itnum < self.iter_newton and
                   np.amax(np.abs(dLAR)) > self.eps_newton):
                itnum += 1

                # -------- compute next residual
                lar_list = self.develop_parcor(LAR.ravel(), basis)
                parcor_list = self.decode(lar_list)
                forward_res_next, backward_res_next = self.cell(
                    parcor_list, self.forward_residual[k],
                    self.backward_residual[k])
                self.forward_residual[k + 1] = forward_res_next
                self.backward_residual[k + 1] = backward_res_next

                # -------- correct the current vector
                g = self.common_gradient(k + 1, parcor_list)
                # n_epochs, n_points - 1 = g.shape = h.shape
                # n_basis, n_epochs, n_points = basis.shape
                gradient = 2.0 * np.dot(
                    basis[:, :, 1:n_points].reshape(n_basis, -1),
                    g.reshape(-1, 1))
                gradient.shape = (n_basis, 1)

                h = self.common_hessian(k + 1, parcor_list)
                hessian = 2.0 * np.dot(
                    (basis[:, :, 1:n_points] * h).reshape(n_basis, -1),
                    basis[:, :, 1:n_points].reshape(n_basis, -1).T)

                dLAR = np.reshape(
                    linalg.solve(hessian, gradient), (1, n_basis))
                LAR -= dLAR

            # -------- save current cell and residuals
            lar_list = self.develop_parcor(LAR.ravel(), basis)
            parcor_list = self.decode(lar_list)

            forward_res, backward_res = self.cell(parcor_list,
                                                  self.forward_residual[k],
                                                  self.backward_residual[k])
            self.forward_residual[k + 1] = forward_res
            self.backward_residual[k + 1] = backward_res

            AR_ = np.vstack((AR_, np.reshape(LAR, (1, n_basis))))
            yield AR_

    def estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_

        """
        if not recompute:
            # if residual are stored, simply return the right one
            try:
                self.residual_ = self.forward_residual[self.ordar_]
            # otherwise, compute the prediction error
            except AttributeError:
                recompute = True

        if recompute:
            self.residual_, _ = self.whiten()

    def develop(self, basis):
        """Compute the AR models and gains at instants fixed by newcols

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1 + ordar, n_epochs, n_columns)
        The size of Gcols is (1, n_epochs, n_columns)

        """
        n_basis, n_epochs, n_points = basis.shape
        ordar = self.ordar_

        # -------- expand on the basis
        AR_cols = np.ones((1, n_epochs, n_points))
        if ordar > 0:
            parcor_list = self.develop_parcor(self.AR_, basis)
            parcor_list = self.decode(parcor_list)
            AR_cols = np.vstack((AR_cols, ki2ai(parcor_list)))
        G_cols = self.develop_gain(basis, squared=False, log=False)

        return AR_cols, G_cols
