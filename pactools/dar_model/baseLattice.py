from __future__ import print_function
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg

from .baseAR import BaseAR
from ..utils.arma import ki2ai


class BaseLattice(BaseAR):
    __metaclass__ = ABCMeta

    def __init__(self, iter_newton=0, eps_newton=0.001, **kwargs):
        """Creates a base Lattice model with Taylor expansion

        iter_newton : maximum number of Newton-Raphson iterations
        eps_newton  : threshold to stop Newton-Raphson iterations

        Other parameters for base AR (super class) are:

        ordar     : order of the autoregressive model
        ordriv    : order of the taylor expansion
        bic       : select order through BIC [boolean]
        normalize : normalize basis (to unit energy) [boolean]
        ortho     : orthogonalize basis [boolean]
        center    : subtract mean from signal [boolean]
        fmax      : highest frequency to plot
        """
        super(BaseLattice, self).__init__(**kwargs)
        self.iter_newton = iter_newton
        self.eps_newton = eps_newton

    def synthesis(self, sigdriv=None, sigin_init=None):
        """Create a signal from fitted model

        sigdriv      : driving signal
        sigdriv_init : shape (1, self.ordar_) previous point of the past

        """
        ordar_ = self.get_ordar()
        if sigdriv is None:
            sigdriv = self.sigdriv
            try:
                newbasis = self.basis_
            except:
                newbasis = self.make_basis()
        else:
            newbasis = self.make_basis(sigdriv=sigdriv)

        _, tmax = sigdriv.shape
        tmax = tmax - ordar_
        if sigin_init is None:
            sigin_init = np.random.randn(1, ordar_)
        sigin_init = np.atleast_2d(sigin_init)
        if sigin_init.shape[1] < ordar_:
            raise(ValueError, "initialization of incorrect length: %d instead"
                              "of %d (self.ordar_)" %
                              (sigin_init.shape[1], ordar_))

        # initialization
        signal = np.zeros(tmax + ordar_)
        signal[:ordar_] = sigin_init[0, -ordar_:]

        # compute basis corresponding to sigdriv
        newbasis = self.make_basis(sigdriv=sigdriv)

        # synthesis of the signal
        random_sig = np.random.randn(1, tmax)
        burn_in = np.random.randn(1, ordar_)
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
        ordar = self.get_ordar()

        # -------- prepare excitation signal and burn in phase
        if burn_in is None:
            tburn = 0
            e = random_sig
        else:
            tburn = burn_in.size
            e = np.hstack((burn_in, random_sig))

        # -------- allocate arrays for forward and backward residuals
        e_forward = np.zeros(ordar + 1)
        e_backward = np.zeros(ordar + 1)

        # -------- prepare parcor coefficients, shape = (ordar, tmax)
        parcor = self.decode(np.dot(self.AR_, basis))
        gain = np.exp(np.dot(self.G_, basis))

        # -------- create the output signal
        tmax = random_sig.size
        sigout = np.zeros(tmax)
        for t in range(-tburn, tmax):
            e_forward[ordar] = e[0, t + tburn] * gain[0, max(t, 0)]
            for p in range(ordar, 0, -1):
                e_forward[p - 1] = (e_forward[p]
                                    - parcor[p - 1, max(t, 0)]
                                    * e_backward[p - 1])
            for p in range(ordar, 0, -1):
                e_backward[p] = (e_backward[p - 1]
                                 + parcor[p - 1, max(t, 0)]
                                 * e_forward[p - 1])
            e_backward[0] = e_forward[0]
            sigout[max(t, 0)] = e_forward[0]
        sigout.shape = (1, tmax)

        return sigout

    def cell(self, parcor, forward_residual, backward_residual, tmax=None):
        """Apply a single cell of the direct lattice filter
        to whiten the original signal

        parcor            : array, lattice coefficients
        forward_residual  : array, forward residual from previous cell
        backward_residual : array, backward residual from previous cell

        the input arrays should have same shape as self.sigin

        returns:
        forward_residual  : forward residual after this cell
        backward_residual : backward residual after this cell

        """
        if tmax is None:
            tmax = self.sigin.size
        f_residual = np.copy(forward_residual)
        b_residual = np.zeros(backward_residual.shape)
        delayed = np.copy(backward_residual[:1, 0:tmax - 1])
        delayed.shape = (1, tmax - 1)

        # -------- apply the cell
        b_residual[:1, 1:tmax] = delayed + (parcor[:1, 1:tmax]
                                            * f_residual[:1, 1:tmax])
        b_residual[0, 0] = parcor[0, 0] * f_residual[0, 0]
        f_residual[:1, 1:tmax] += parcor[:1, 1:tmax] * delayed
        return f_residual, b_residual

    def whiten(self):
        """Apply the direct lattice filter to whiten the original signal

        returns:
        residual : array containing the residual (whitened) signal
        backward : array containing the backward residual (whitened) signal

        """
        # -------- crop the beginning of the signal
        sigin = self.crop_begin(self.sigin)
        basis = self.crop_begin(self.basis_)

        # -------- select ordar (prefered: ordar_)
        ordar = self.get_ordar()
        tmax = sigin.size

        residual = np.copy(sigin)
        backward = np.copy(sigin)
        for k in range(ordar):
            parcor = np.reshape(np.dot(self.AR_[k], basis), (1, tmax))
            parcor = self.decode(parcor)
            residual, backward = self.cell(parcor, residual, backward, tmax)

        return residual, backward

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
        g : array containing the factors (size (1, tmax - 1))

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
        h : array containing the factors (size (1, tmax - 1))

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

        # -------- crop the end of the signal, to fit only on the beginning
        sigin = self.crop_end(self.sigin)
        basis = self.crop_end(self.basis_)

        # -------- select signal, basis and regression signals
        _, tmax = sigin.shape
        ordriv_p1 = self.ordriv + 1
        ordar_ = self.ordar
        scale = 1.0 / tmax

        # -------- prepare residual signals
        forward_res = np.copy(sigin)
        backward_res = np.copy(sigin)

        # -------- model at order 0
        AR_ = np.empty((0, ordriv_p1))
        self.forward_residual = np.empty((ordar_ + 1, tmax))
        self.backward_residual = np.empty((ordar_ + 1, tmax))
        self.forward_residual[0] = forward_res
        self.backward_residual[0] = backward_res
        yield AR_

        # -------- loop on successive orders
        for k in range(0, ordar_):
            # -------- prepare initial estimation (driven parcor)
            forward_reg = basis * forward_res
            backward_reg = basis * backward_res
            R = scale * (np.dot(forward_reg[0:, 1:],
                                forward_reg[0:, 1:].T)
                         + np.dot(backward_reg[0:, :-1],
                                  backward_reg[0:, :-1].T))
            r = (scale * 2.0) * np.dot(forward_reg[0:, 1:],
                                       backward_res[:1, :-1].T)
            parcor = -np.linalg.solve(R, r).T
            parcor_list = np.dot(parcor, basis)
            parcor_list = np.maximum(parcor_list, -0.999999)
            parcor_list = np.minimum(parcor_list, 0.999999)

            R = scale * np.dot(basis, basis.T)
            r = scale * np.dot(basis, self.encode(parcor_list).T)
            LAR = np.linalg.solve(R, r).T

            # -------- Newton-Raphson refinement
            dLAR = np.inf
            itnum = 0
            while (itnum < self.iter_newton
                   and np.amax(np.abs(dLAR)) > self.eps_newton):
                itnum += 1

                # -------- compute next residual
                lar_list = np.dot(LAR, basis)
                parcor_list = self.decode(lar_list)
                forward_res_next, backward_res_next = self.cell(
                    parcor_list, forward_res, backward_res, tmax)
                self.forward_residual[k + 1] = forward_res_next
                self.backward_residual[k + 1] = backward_res_next

                # -------- correct the current vector
                g = self.common_gradient(k + 1, parcor_list).T
                gradient = 2.0 * np.dot(basis[:, 1:tmax], g)
                gradient.shape = (self.ordriv + 1, 1)
                h = self.common_hessian(k + 1, parcor_list)
                hessian = 2.0 * np.dot(basis[:, 1:tmax] * h,
                                       basis[:, 1:tmax].T)

                dLAR = np.reshape(linalg.solve(hessian, gradient),
                                  (1, self.ordriv + 1))
                LAR -= dLAR

            # -------- save current cell and residuals
            lar_list = np.dot(LAR, basis)
            parcor_list = self.decode(lar_list)

            forward_res, backward_res = self.cell(
                parcor_list, forward_res, backward_res, tmax)
            self.forward_residual[k + 1] = forward_res
            self.backward_residual[k + 1] = backward_res

            AR_ = np.vstack((AR_, np.reshape(LAR, (1, ordriv_p1))))
            yield AR_

    def estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_

        """
        # if we fit only on the beginning, we need to recompute the residual
        if self.fit_size < 1:
            recompute = True

        # -------- crop the beginning of the signal
        sigin = self.crop_begin(self.sigin)
        tmax = sigin.size

        if not recompute:
            # -------- if residual are stored, simply return the right one
            try:
                self.residual_ = np.reshape(self.forward_residual[self.ordar_],
                                            (1, tmax))
            # -------- otherwise, compute the prediction error
            except AttributeError:
                recompute = True

        if recompute:
            self.residual_, _ = self.whiten()

    def develop(self, newcols, newbasis=None):
        """Compute the AR models and gains at instants fixed by newcols

        newcols : array giving the indexes of the columns
        newbasis : if None, we use the basis used for fitting (self.basis_)
                    else, we use the given basis.
        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1+ordar, len(newcols))
        The size of Gcols is (1, len(newcols))

        """
        if newbasis is None:
            basisplot = self.basis_[:, newcols]
        else:
            basisplot = newbasis[:, newcols]
        _, nbcols = basisplot.shape
        ordar = self.get_ordar()

        # -------- expand on the basis
        AR_cols = np.ones((1, nbcols))
        if ordar > 0:
            ki = np.dot(self.AR_, basisplot)
            ki = self.decode(ki)
            AR_cols = np.vstack((AR_cols,
                                 ki2ai(ki)))
        G_cols = np.exp(np.dot(self.G_, basisplot))
        return (AR_cols, G_cols)
