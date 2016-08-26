from __future__ import print_function
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import stats

from ..utils.progress_bar import ProgressBar
from ..utils.maths import squared_norm
from ..utils.spectrum import phase_amplitude


class BaseAR(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 ordar=1,
                 ordriv=0,
                 bic=False,
                 normalize=True,
                 ortho=True,
                 center=True,
                 iter_gain=10,
                 epsilon=1.0e-4,
                 iter_newton=0,
                 eps_newton=0.001,
                 ordriv_d=0,
                 progress_bar=False):
        """Creates a filtered STAR model with Taylor expansion

        ordar      : order of the autoregressive model
        ordriv     : order of the taylor expansion
        bic        : select order through BIC [boolean]
        normalize  : normalize basis (to unit energy) [boolean]
        ortho      : orthogonalize basis [boolean]
        center     : subtract mean from signal [boolean]
        iter_gain  : maximum number of iteration in gain estimation
        epsilon    : threshold to stop iterations in gain estimation
        iter_newton : maximum number of Newton-Raphson iterations
        eps_newton  : threshold to stop Newton-Raphson iterations
        ordriv_d   : ordriv for driver's derivative

        """
        # -------- save parameters
        self.ordar = ordar          # autoregressive order
        self.bic = bic              # find orders through BIC
        self.normalize = normalize  # normalize the amplitude of driving signal
        self.ortho = ortho          # Gram-Schmidt orthogonalization of basis
        self.center = center        # subtract mean from sigin
        self.iter_gain = iter_gain
        self.epsilon = epsilon
        self.progress_bar = progress_bar

        # power of the Taylor expansion
        self.ordriv = ordriv + ordriv_d
        self.ordriv_d = ordriv_d

        # -------- prepare other arrays
        self.basis_ = None
        self.fit_size = 1

    def fit(self, sigin, sigdriv, fs, mask=None,
            fit_size=1, use_driver_phase=False):
        """Estimates a filtered STAR model with Taylor expansion

        sigin     : signal that is to be modeled
        sigdriv   : signal that drives the model
        fs        : sampling frequency
        mask      : mask to select where signals are used to fit the model
        fit_size         : ratio of the signal used in the fit
        use_driver_phase : if True, we divide the driver by its amplitude

        start, filein and filedriv are only given for information
        and may be omitted
        """
        self.reset_logl_aic_bic()
        self.use_driver_phase = use_driver_phase
        if use_driver_phase:
            phase, _ = phase_amplitude(sigdriv, phase=True, amplitude=False)
            sigdriv = np.cos(phase)
            # phase_amplitude may crop to speed up the Hilbert transform
            sigin = np.atleast_2d(sigin)[:, :sigdriv.size]

        # -------- save signals as attributes of the model
        self.sigin = np.atleast_2d(np.array(sigin, dtype='float64'))
        self.sigdriv = np.atleast_2d(np.array(sigdriv, dtype='float64'))

        self.mask = mask
        if mask is not None:
            self.mask = np.atleast_2d(np.array(mask, dtype='float64'))

        if self.center:
            self.sigin -= np.mean(self.sigin)

        if self.sigin.shape != self.sigdriv.shape:
            raise ValueError(
                'sigin and sigdriv have incompatible shapes. Got '
                '(%s != %s)' % (self.sigin.shape, self.sigdriv.shape))

        # -------- informations on the signals
        self.fs = fs
        self.fit_size = fit_size

        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)

        # -------- estimation of the model
        if self.bic:
            # -------- select the best order
            self.AIC, self.BIC, self.logL, self.tmax = self.order_selection()
            self.estimate_error()
            self.estimate_gain()

        else:
            # -------- estimate a single model
            self.make_basis()
            self.estimate_ar()
            self.estimate_error()
            self.estimate_gain()
        return self

    def make_basis(self, sigdriv=None):
        """Creates a basis from the driving signal

        sigdriv : if None, we use self.sigdriv instead

        the basis is build with the successive powers of self.sigdriv

        if self.normalize is True, each component of the basis will be
        normalized by its standard deviation
        if self.ortho is True, a Gram-Schmidt orthogonalisation of the
        basis will be performed

        the basis is stored in self.basis_
        """
        ordriv, ordriv_d = self.get_ordriv(), self.get_ordriv_d()
        ordriv -= ordriv_d
        # -------- check default arguments
        if sigdriv is None:
            sigdriv = self.sigdriv
            save_basis = True
            ortho = self.ortho
            normalize = self.normalize
        else:
            # reuse previous transformation
            save_basis = False
            ortho = False
            normalize = False

        sigdriv = np.atleast_2d(sigdriv)
        n_epochs, n_points = sigdriv.shape

        # -------- memorize in alpha the various transforms
        alpha = np.eye(ordriv + ordriv_d + 1)

        # -------- create the basis
        basis = np.empty((ordriv + ordriv_d + 1, n_epochs * n_points))
        basis[0] = 1.0

        # -------- create derivative
        if ordriv_d > 0:
            sigdriv_d = np.copy(sigdriv)
            sigdriv_d -= np.roll(sigdriv, shift=1, axis=1)
            sigdriv_d[:, 0] = sigdriv_d[:, 1]
            sigdriv_d *= sigdriv.std() / sigdriv_d.std()

        orders = np.r_[np.arange(1, ordriv + 1), np.arange(1, ordriv_d + 1)]
        derivative = [False] * ordriv + [True] * ordriv_d
        # -------- create successive components
        for k, (order, deriv) in enumerate(zip(orders, derivative)):
            k += 1
            if deriv:
                basis[k] = sigdriv_d.ravel() ** order
            else:
                basis[k] = sigdriv.ravel() ** order

            if ortho:
                # correct current component with corrected components
                for m in range(k):
                    alpha[k, m] = - (np.sum(basis[k] * basis[m]) /
                                     np.sum(basis[m] * basis[m]))
                basis[k] += np.dot(alpha[k, :k], basis[:k, :])
                # recompute the expression over initial components
                alpha[k, :k] = np.dot(alpha[k, :k], alpha[:k, :k])
            if normalize:
                scale = np.sqrt(float(n_epochs * n_points) /
                                squared_norm(basis[k]))
                basis[k] *= scale
                alpha[k, :k + 1] *= scale

        # -------- save basis and transformation matrix
        # (basis_ = transform_ * rawbasis)
        if save_basis:
            self.basis_ = basis.reshape(-1, n_epochs, n_points)
            self.transform_ = alpha
        else:
            if self.normalize or self.ortho:
                basis = np.dot(self.transform_, basis)
            return basis.reshape(-1, n_epochs, n_points)

    def estimate_ar(self):
        """Estimates the AR model on a signal

        uses self.sigin, self.sigdriv and self.basis_ (which must be
        computed with make_basis before calling estimate_ar)
        self.ordar defines the order
        self.ordar_ receives this value after estimation
        """

        # -------- create the observation and regression signals
        # sigin        : input signal (copy of self.sigin)
        # sigdriv      : driving signal (copy of self.sigdriv)
        # basis_       : basis of functions (build from sigdriv)
        # sigreg       : regression signals (sigin * basis)

        # -------- verify all parameters for the estimator
        if self.ordar < 1:
            raise ValueError('baseAR: self.ordar is zero or negative')
        if self.ordriv < 0:
            raise ValueError('baseAR: self.ordriv is negative')
        if self.basis_ is None:
            raise ValueError('baseAR: basis does not yet exist')

        # -------- compute the final estimate
        if self.progress_bar:
            bar = ProgressBar(title=self.get_title(name=True))
        for AR_ in self.last_model():
            self.AR_ = AR_
            self.ordar_ = AR_.shape[0]
            if self.progress_bar:
                bar.update(float(self.ordar_) / self.ordar,
                           title=self.get_title(name=True))
        self.ordriv_ = self.ordriv

    def crop_end(self, sig):
        fit_size = min(1, self.fit_size)
        if fit_size == 1:
            return sig
        else:
            n_points = sig.shape[-1]
            n_points = int(n_points * fit_size)
            return sig[..., :n_points]

    def crop_begin(self, sig):
        fit_size = min(1, self.fit_size)
        if fit_size == 1:
            return sig
        else:
            n_points = sig.shape[-1]
            n_points = int(n_points * fit_size)
            return sig[..., n_points:]

    def degrees_of_freedom(self):
        return ((self.get_ordar() + 1) * (self.get_ordriv() + 1))

    def get_title(self, name=False, logl=False, bic=False):
        title = ''
        if name:
            title += self.__class__.__name__

        ordar = self.get_ordar()
        ordriv = self.get_ordriv() - self.get_ordriv_d()
        ordriv_d = self.get_ordriv_d()
        if ordriv_d > 0:
            title += '(%d, %d+%d)' % (ordar, ordriv, ordriv_d)
        else:
            title += '(%d, %d)' % (ordar, ordriv)

        if logl:
            logL = self.get_logl()
            title += 'logL=%.4f' % logL
        if bic:
            bic = self.get_bic()
            title += '_BIC=%.4f' % bic
        return title

    def get_ordar(self):
        # -------- distinguish estimated or target model
        try:
            return self.ordar_
        except AttributeError:
            return self.ordar

    def get_ordriv(self):
        # -------- distinguish estimated or target model
        try:
            return self.ordriv_
        except AttributeError:
            return self.ordriv

    def get_ordriv_d(self):
        # -------- distinguish estimated or target model
        try:
            return self.ordriv_d_
        except AttributeError:
            return self.ordriv_d

    def get_full_criterion(model, criterion):
        if criterion == 'AIC':
            array = model.AIC
        elif criterion == 'BIC':
            array = model.BIC
        elif criterion == 'logL':
            array = model.LogL

        if not isinstance(array, np.ndarray):
            raise ValueError('get_full_criterion gives the full '
                             'array of model selection, yet no model selection'
                             ' has been done.')
        return array

    def get_criterion(self, criterion):
        if criterion == 'AIC':
            value = self.get_aic()
        elif criterion == 'BIC':
            value = self.get_bic()
        elif criterion == 'logL':
            value = self.get_logl()
        elif criterion == '-logL':
            value = -self.get_logl()
        else:
            raise ValueError('Wrong criterion: %s' % criterion)
        return value

    def get_bic(self):
        BIC = getattr(self, 'BIC', None)
        if BIC is None:
            logL = self.get_logl()
            tmax = getattr(self, 'tmax', None)

            eta_bic = np.log(tmax)
            BIC = -2.0 * logL + eta_bic * self.degrees_of_freedom() / tmax
            self.BIC = BIC

        if isinstance(BIC, np.ndarray):
            BIC = BIC[self.ordar_, self.ordriv_ - self.ordriv_d_,
                      self.ordriv_d_]

        return BIC

    def get_aic(self):
        AIC = getattr(self, 'AIC', None)
        if AIC is None:
            logL = self.get_logl()
            tmax = getattr(self, 'tmax', None)

            eta_aic = 2.0
            AIC = -2.0 * logL + eta_aic * self.degrees_of_freedom() / tmax
            self.AIC = AIC

        if isinstance(AIC, np.ndarray):
            AIC = AIC[self.ordar_, self.ordriv_ - self.ordriv_d_,
                      self.ordriv_d_]

        return AIC

    def get_logl(self):
        logL = getattr(self, 'logL', None)
        if logL is None:
            logL, tmax = self.log_likelihood(skip=self.ordar_)
            logL /= tmax
            self.logL = logL
            self.tmax = tmax

        if isinstance(logL, np.ndarray):
            logL = logL[self.ordar_, self.ordriv_ - self.ordriv_d_,
                        self.ordriv_d_]

        return logL

    def reset_logl_aic_bic(self):
        self.logL = None
        self.BIC = None
        self.AIC = None
        self.tmax = None

    def order_selection(self):
        """Select the order of the model

        Returns the values of the criterion (a tuple containing the
        AIC and BIC criterions, and the log likelihood)

        Attributes ordar and ordriv define the maximum values.
        This function saves the optimal values as ordar_ and ordriv_
        """
        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)
        ordar = self.ordar
        ordriv = self.ordriv - self.ordriv_d
        ordriv_d = self.ordriv_d

        best_BIC = np.inf
        logL = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))
        AIC = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))
        BIC = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))

        # -------- loop on ordriv with a copy of the estimator
        if self.progress_bar:
            bar = ProgressBar(title='%s' % self.__class__.__name__,
                              max_value=logL.size)
            bar_k = 0
        for ordriv_d_ in range(ordriv_d + 1):
            for ordriv_ in range(ordriv + 1):
                A = self.copy()
                A.ordriv = ordriv_ + ordriv_d_
                A.ordriv_ = ordriv_ + ordriv_d_
                A.ordriv_d = ordriv_d_
                A.ordriv_d_ = ordriv_d_
                A.make_basis()

                # -------- estimate the best AR order for this value of ordriv
                for AR_ in A.next_model():
                    ordar_ = AR_.shape[0]
                    A.AR_ = AR_
                    A.ordar_ = ordar_
                    if self.progress_bar:
                        title = A.get_title(name=True)
                        bar_k += 1
                        bar.update(bar_k, title=title)
                    A.estimate_error()
                    A.estimate_gain()

                    A.reset_logl_aic_bic()
                    this_BIC = A.get_bic()
                    logL[ordar_, ordriv_, ordriv_d_] = A.get_logl()
                    AIC[ordar_, ordriv_, ordriv_d_] = A.get_aic()
                    BIC[ordar_, ordriv_, ordriv_d_] = this_BIC

                    # -------- actualize the best model
                    if this_BIC < best_BIC:
                        best_BIC = this_BIC
                        self.ordar_ = ordar_
                        self.ordriv_ = ordriv_ + ordriv_d_
                        self.ordriv_d_ = ordriv_d_
                        self.AR_ = np.copy(AR_)
                        self.G_ = np.copy(A.G_)

        # -------- return criteria
        self.make_basis()
        tmax = A.tmax + ordar - self.ordar_
        return (AIC, BIC, logL, tmax)

    def estimate_gain(self, regul=0.01):
        """Estimate the gain expressed over the basis by maximizing
        the likelihood, through a Newton-Raphson procedure

        The residual e(t) is modelled as a white noise with standard
        deviation: std(e(t)) = exp(b(0) + b(1)u(t) + ... + b(m)u(t)^M)

        iter_gain : maximum number of iterations
        epsilon   : stop iteration when corrections get smaller than epsilon
        regul     : regularization factor (for inversion of the Hessian)
        """
        # -------- estimate the gain (self.G_) given the following data
        # self.ordriv    : order of the regression of the parameters
        # self.residual_ : residual signal (from ordar to n_points)
        # self.basis_    : basis of functions (build from sigdriv)

        # -------- compute these intermediate quantities
        # resreg    : regression residual (residual * basis)
        # residual  : copy of self.residual_
        # residual2 : squared residual

        iter_gain = self.iter_gain
        epsilon = self.epsilon
        ordar = self.get_ordar()

        # -------- crop the beginning of the signal (for left out data)
        basis = self.crop_begin(self.basis_)
        mask = self.crop_begin(self.mask) if self.mask is not None else None

        residual = self.residual_

        # -------- crop the ordar first values
        residual = residual[:, ordar:]
        basis = basis[:, :, ordar:]
        if mask is not None:
            mask = mask[:, ordar:]

        # concatenate the epochs since it does not change the computations
        # as in estimating the AR coefficients
        if mask is not None:
            residual = residual[mask != 0]
            basis = basis[:, mask != 0]
            mask = mask[mask != 0].reshape(1, -1)
        residual = residual.reshape(1, -1)
        basis = basis.reshape(basis.shape[0], -1)

        residual2 = residual ** 2
        n_points = residual.size
        resreg = basis * residual
        self.G_ = np.zeros((1, self.ordriv_ + 1))

        # -------- prepare an initial estimation
        # chose N = 3 * ordriv classes, estimate a standard deviation
        # on each class, and make a regression on the values of
        # the indexes for their centroid.
        # For a time-dependent signal, a good idea was to take
        # successive slices of the signal, but for signal driven
        # models, this would be wrong! Use levels of the driving
        # function instead.
        if self.ordriv_ > 0:
            index = np.argsort(basis[1, :])  # indexes to sort the basis
        else:
            index = np.arange(n_points, dtype=int)

        nbslices = 3 * (self.ordriv_ + 1)  # number of slices
        lenslice = n_points // nbslices    # length of a slice
        e = np.zeros(nbslices)             # log energies

        # -------- prepare least-squares equations
        tmp = lenslice * np.arange(nbslices + 1)
        kmin = tmp[:-1]
        kmax = tmp[1:]
        kmid = (kmin + kmax) // 2
        for k, (this_min, this_max) in enumerate(zip(kmin, kmax)):
            e[k] = np.mean(residual2[0, index[this_min:this_max]])

        if mask is not None:
            masked_basis = basis * mask
        else:
            masked_basis = basis

        e = 0.5 * np.log(e)
        R = np.dot(basis[:, index[kmid]],
                   masked_basis[:, index[kmid]].T)
        r = np.dot(e, masked_basis[:, index[kmid]].T)

        # -------- regularize matrix R
        v = np.resize(linalg.eigvalsh(R), (1, self.ordriv + 1))
        correction = regul * np.max(v)

        # add tiny correction on diagonal without a copy
        R.flat[::len(R) + 1] += correction

        # -------- compute regularized solution
        self.G_ = linalg.solve(R, r)
        self.G_.shape = (1, self.ordriv_ + 1)

        # -------- prepare the likelihood (and its initial value)
        loglike = np.zeros(iter_gain + 1)
        loglike.fill(np.nan)

        # -------- refine this model (iteratively maximise the likelihood)
        for itnum in range(iter_gain):
            logsigma = np.dot(self.G_, basis)
            sigma2 = np.exp(2 * logsigma)

            # TODO: add weighting if mask is not None
            loglike[itnum] = wgn_log_likelihood(residual, sigma2)

            gradient = np.sum(basis * (residual2 / sigma2 - 1.0), 1)
            hessian = -2.0 * np.dot(resreg / sigma2, resreg.T)
            dG = linalg.solve(hessian, gradient)
            self.G_ -= dG.T
            if np.amax(np.absolute(dG)) < epsilon:
                iter_gain = itnum + 1
                break
        self.residual_bis_ = residual / np.sqrt(sigma2)

    def develop_gain(self, basis, sigdriv=None, squared=False, log=False):
        # n_basis, n_epochs, n_points = basis.shape
        # 1, n_basis = self.G_.shape
        # n_epochs, n_points = gain.shape
        G_cols = np.tensordot(self.G_.ravel(), basis, axes=([0], [0]))
        if squared:
            G_cols *= 2
        if not log:
            G_cols = np.exp(G_cols)
        return G_cols

    def log_likelihood(self, skip=0):
        """Approximate computation of the logarithm of the likelihood
        the computation uses self.residual_, self.basis_ and self.G_

        skip : how many initial samples to skip
        """
        # -------- crop the beginning of the signal
        basis = self.crop_begin(self.basis_)

        # -------- estimate the gain
        gain2 = self.develop_gain(basis[:, :, skip:], squared=True)

        # -------- compute the log likelihood from the residual
        logL = wgn_log_likelihood(self.residual_[:, skip:], gain2)

        tmax = gain2.size

        return logL, tmax

    def likelihood_ratio(self, ar0):
        """Computation of the likelihood ratio test
        the null hypothesis H0 is given by model ar0; the current
        model (self) is tested against this model ar0 which may be
        either an invariant AR model (ordriv=0) or a hybrid one
        (invariant AR part, with driven gain).

        ar0 : model representing the null hypothesis

        returns:
        pvalue : the p-value for the test (1-F(chi2(ratio)))
        """
        # -------- crop the beginning of the signal
        sigdriv = self.sigdriv
        sigin = self.sigin

        # -------- verify that the current model has been fitted
        if not hasattr(self, 'ordar_'):
            raise ValueError('model was not fit before LR test')

        # -------- verify that the H0 model has been fitted, or fit it
        if not hasattr(ar0, 'ordar_'):
            ar0.fit(sigin, sigdriv, fs=self.fs)

        # -------- compute the log likelihood for the current model
        logL1, _ = self.log_likelihood(skip=self.ordar_)

        # -------- compute the log likelihood for the H0 model
        logL0, _ = ar0.log_likelihood(skip=ar0.ordar_)

        # -------- compute the log likelihood ratio
        test = 2.0 * (logL1 - logL0)

        # -------- compute degrees of freedom
        df1 = self.degrees_of_freedom()
        df0 = ar0.degrees_of_freedom()

        # -------- compute the p-value
        pvalue = stats.chi2.sf(test, df1 - df0)

        return pvalue

    # ------------------------------------------------ #
    # Functions that should be overloaded              #
    # ------------------------------------------------ #
    @abstractmethod
    def next_model(self):
        """Compute the AR model at successive orders

        Acts as a generator that stores the result in self.AR_
        Creates the models with orders from 0 to self.ordar

        Typical usage:
        for AR_ in A.next_model():
            A.AR_ = AR_
            A.ordar_ = AR_.shape[0]
        """
        pass

    @abstractmethod
    def estimate_error(self):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_
        """
        pass

    @abstractmethod
    def develop(self, newcols, newbasis):
        """Compute the AR models and gains at instants fixed by newcols

        newcols : array giving the indexes of the columns

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1+ordar, len(newcols))
        The size of Gcols is (1, len(newcols))
        """
        pass

    def last_model(self):
        """overload if it's faster to get only last model"""
        return self.next_model()

    def develop_all(self, sigdriv=None, instantaneous=False):
        """
        Compute the AR models and gains for every instant of sigdriv
        """

        if sigdriv is None:
            sigdriv = self.sigdriv
            basis = self.make_basis()
        else:
            basis = self.make_basis(sigdriv=sigdriv)

        newcols = np.arange(sigdriv.size)
        try:
            AR_cols, G_cols = self.develop(newcols=newcols, newbasis=basis,
                                           instantaneous=instantaneous)
        except:
            AR_cols, G_cols = self.develop(newcols=newcols, newbasis=basis)

        return AR_cols, G_cols, newcols, sigdriv

    # ------------------------------------------------ #
    # Functions to plot the models                     #
    # ------------------------------------------------ #
    def _basis2spec(self, newcols, frange=None, instantaneous=False,
                    sigdriv=None):
        """Compute the power spectral density for a given basis

        newcols : array giving the indexes of the columns
        frange  : frequency range

        this method is not intended for general use, except as a
        factorisation of the code of methods plot_time_freq and
        plot_drive_freq
        """
        # -------- distinguish estimated or target model
        ordar = self.get_ordar()

        # -------- estimate AR and Gain columns
        if sigdriv is not None:
            AR_cols, G_cols, _, _ = self.develop_all(
                sigdriv=sigdriv, instantaneous=instantaneous)
        else:
            try:
                AR_cols, G_cols = self.develop(newcols,
                                               instantaneous=instantaneous)
            except:
                AR_cols, G_cols = self.develop(newcols)

        # -------- estimate AR spectrum
        nfft = 256
        while nfft < (ordar + 1):
            nfft *= 2

        if ordar > 0:
            AR_cols = AR_cols / G_cols
            AR_spec = np.fft.rfft(AR_cols, n=nfft, axis=0)
        else:
            Ginv_cols = 1.0 / G_cols
            AR_spec = np.fft.rfft(Ginv_cols, n=nfft, axis=0)

        # -------- estimate AR spectrum
        spec = -20.0 * np.log10(np.abs(AR_spec))

        # -------- truncate to frange
        if frange is not None:
            frequencies = np.linspace(0, self.fs // 2, 1 + nfft // 2)
            mask = np.logical_and(frequencies <= frange[1],
                                  frequencies > frange[0])
            spec = spec[mask, :]
        return spec

    def amplitude_frequency(self, nbcols=256, frange=None, mode='',
                            xlim=None, fmax=None):
        """Computes an amplitude-frequency power spectral density

        nbcols : number of expected columns (amplitude)
        frange : frequency range
        mode   : normalisation mode ('c' = centered, 'v' = unit variance)
        xlim   : minimum and maximum amplitude

        returns:
        spec : ndarray containing the time-frequency psd
        xlim : minimum and maximum amplitude

        """
        # -------- define the horizontal axis
        if xlim is None:
            xlim = self.get_sigdriv_bounds()

        # simple ramp
        eps = 0.0
        amplitudes = np.linspace(xlim[0] + eps, xlim[1] - eps, nbcols)

        # -------- compute spectra
        spec = self._basis2spec(None, frange, True, amplitudes[None, :])

        # keep the only epoch
        spec = spec[:, 0, :]

        # -------- normalize
        if 'c' in mode:
            spec -= np.mean(spec, axis=1)
        if 'v' in mode:
            spec /= np.std(spec, axis=1)

        return spec, xlim

    def develop_coefficients(self, nbcols=256, xlim=None):
        """Computes the coefficients on the basis

        nbcols : number of expected columns (amplitude)
        xlim : minimum and maximum amplitude

        returns:
        spec : ndarray containing the time-frequency psd
        xlim : minimum and maximum amplitude

        """
        # -------- define the horizontal axis
        if xlim is None:
            xlim = self.get_sigdriv_bounds()

        # simple ramp
        eps = 0.0
        amplitudes = np.linspace(xlim[0] + eps, xlim[1] - eps, nbcols)

        # -------- estimate AR and Gain columns
        AR_cols, _, _, _ = self.develop_all(
            sigdriv=amplitudes[None, :], instantaneous=True)

        return AR_cols, xlim

    def get_sigdriv_bounds(self, sigdriv=None):
        if self.use_driver_phase:
            bounds = [-1, 1]

        else:
            if sigdriv is None:
                sigdriv = self.sigdriv
            if self.mask is not None:
                sigdriv = sigdriv[self.mask != 0]
            bounds = np.percentile(sigdriv, [5, 95])
        bound_min = min(-bounds[0], bounds[1])
        bounds = (-bound_min, bound_min)
        return bounds

    def driven_gain(self, draw='rg', fig=None, ylim=None, n_bins=100,
                    bin_residual=True, sigdriv=None, scale='dB',
                    residual=None, title=None):
        """
        Compute the driven gain for n_bins values of the driver amplitude.
        Draw 'r' the residual and/or 'g' the estimated gain.
        """
        if sigdriv is None:
            sigdriv = self.sigdriv
        sigdriv = self.crop_begin(np.atleast_2d(sigdriv))

        if self.mask is not None:
            mask = self.crop_begin(self.mask)
            sigdriv = sigdriv[mask != 0]

        sigdriv = sigdriv.ravel()

        if not bin_residual or 'r' not in draw:
            # uniform binning
            bounds = self.get_sigdriv_bounds(sigdriv)
            sigdriv_bins = np.linspace(bounds[0], bounds[1], n_bins + 1)

        else:
            # non-uniform binning
            sigdriv_bins = np.percentile(sigdriv,
                                         np.linspace(5, 95, n_bins + 1))

        # compute the G_ coefficients for each instant
        newbasis = self.make_basis(sigdriv=sigdriv_bins[None, :])
        if scale == 'dB':
            G_cols_dB = self.develop_gain(newbasis, sigdriv_bins,
                                          squared=True, log=True)
            G_cols_dB *= 20.0 / np.log(10)
        else:
            G_cols_dB = self.develop_gain(newbasis, sigdriv_bins,
                                          squared=True, log=False)

        # bin the raw residual with respect to sigdriv values (sigdriv_bins)
        if 'r' in draw:
            if residual is None:
                residual = self.crop_begin(self.residual_)

            if self.mask is not None:
                residual = residual[mask != 0]

            residual = residual.ravel() ** 2  # with copy

            if not bin_residual:
                # do not bin the residual
                binned_res_mean = residual
            else:
                # bin the residual
                bin_indices = np.digitize(sigdriv, sigdriv_bins) - 1
                binned_res_mean = np.zeros(n_bins)
                for b in range(n_bins):
                    if np.any(bin_indices == b):
                        this_res = residual[bin_indices == b]
                        binned_res_mean[b] = np.mean(this_res)

            if scale == 'dB':
                binned_res_mean = 20.0 * np.log10(binned_res_mean)

        if draw:
            if fig is None:
                fig = plt.figure('driven gain')
            try:
                ax = fig.gca()
            except:
                ax = fig

            if 'g' in draw:
                ax.plot(sigdriv_bins, G_cols_dB[0])
            if 'r' in draw:
                if bin_residual:
                    ax.plot(0.5 * (sigdriv_bins[:-1] + sigdriv_bins[1:]),
                            binned_res_mean,
                            'o', alpha=0.5)
                    # color=ax.lines[-1].get_color())
                else:
                    ax.plot(sigdriv,
                            binned_res_mean,
                            ',', alpha=0.1)

            ax.set_ylabel('Gain (%s)' % scale)
            ax.set_xlabel('Driver amplitude')
            if title is None:
                title = self.get_title(name=True)
            ax.set_title(title)
            ax.grid('on')
            if ylim is not None:
                ax.set_ylim(ylim)

        if 'r' in draw:
            return G_cols_dB, binned_res_mean
        else:
            return G_cols_dB

    # ------------------------------------------------ #
    def to_dict(self):
        """Convert the model attributes to dict"""

        data = self.__dict__
        for k in data.keys():
            if '__' in k or k[0] == '_':
                del(data[k])
        return data

    def copy(self):
        """Creates a (deep) copy of a model"""

        A = self.__class__()
        A.AR_ = np.copy(self.AR_)
        A.G_ = np.copy(self.G_)
        try:
            A.basis_ = np.copy(A.basis_)
        except AttributeError:
            None

        # -------- copy other attributes
        data = self.to_dict()
        for k in data.keys():
            if k not in ('AR_', 'G_') and k[0] != '_':
                setattr(A, k, data[k])
        return A

    def __repr__(self):
        """Represent a model"""

        s = '%s(ordar=%d, ordriv=%d, bic=%s)' % (self.__class__.__name__,
                                                 self.ordar,
                                                 self.ordriv,
                                                 self.bic)
        try:
            s += '.fit(sigin=ndarray%s' % self.sigin.shape
        except AttributeError:
            s += '.fit(sigin=None'
        try:
            s += ', sigdriv=ndarray%s' % self.sigdriv.shape
        except AttributeError:
            s += ', sigdriv=None'
        try:
            s += ', fs=%.f' % self.fs
        except AttributeError:
            s += ', fs=None'
        s += ') '
        return s

    def __str__(self):
        return self.get_title(name=True, logl=False)


def wgn_log_likelihood(eps, sigma2):
    """Returns the likelihood of a gaussian uncorrelated signal
    given a model of its variance

    eps    : gaussian signal (residual or innovation)
    sigma2 : variance of this signal
    """
    tmax = eps.size
    logL = tmax * np.log(2.0 * np.pi)  # = tmax * 1.8378770664093453
    logL += np.sum(eps * eps / sigma2)  # = tmax * 1.0 if fitted correctly
    logL += np.sum(np.log(sigma2))
    logL *= -0.5

    return logL
