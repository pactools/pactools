from __future__ import print_function
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import linalg
from scipy.signal import fftconvolve
from scipy import stats

from ..utils.progress_bar import ProgressBar
from ..utils.maths import squared_norm
from ..utils.spectrum import phase_amplitude


class BaseDAR(object):
    __metaclass__ = ABCMeta

    def __init__(self, ordar=1, ordriv=0, ordriv_d=0, criterion=None,
                 normalize=True, ortho=True, center=True, iter_gain=10,
                 eps_gain=1.0e-4, progress_bar=False,
                 cross_term_driver=True, use_driver_phase=False):
        """
        Parameters
        ----------
        ordar : int > 0
            Order of the autoregressive model

        ordriv : int >= 0
            Order of the taylor expansion for sigdriv

        ordriv_d : int >= 0
            Order of the taylor expansion for sigdriv_imag

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

        cross_term_driver: boolean
            If True, the cross terms are used when using two drivers.

        use_driver_phase : boolean
            If True, we divide the driver by its instantaneous amplitude.

        """
        # -------- save parameters
        self.ordar = ordar
        self.criterion = criterion
        self.normalize = normalize
        self.ortho = ortho
        self.center = center
        self.iter_gain = iter_gain
        self.eps_gain = eps_gain
        self.progress_bar = progress_bar
        self.use_driver_phase = use_driver_phase

        # power of the Taylor expansion
        self.ordriv = ordriv
        self.ordriv_d = ordriv_d
        self.cross_term_driver = cross_term_driver
        self.compute_cross_orders(ordriv, ordriv_d, cross_term_driver)

        # left-out system
        self.train_mask = None
        self.test_mask = None

        # prepare other arrays
        self.basis_ = None
        self.ordar_ = ordar
        self.ordriv_ = ordriv
        self.ordriv_d_ = ordriv_d

    def check_all_arrays(self, sigin, sigdriv, sigdriv_imag,
                         train_mask, test_mask):
        # -------- transform the signals to 2d array of float64
        sigin = check_array(sigin)
        sigdriv = check_array(sigdriv)
        check_consistent_shape(sigin, sigdriv)

        # -------- also store the sigdriv imaginary part if present
        if sigdriv_imag is not None:
            sigdriv_imag = check_array(sigdriv_imag)
            check_consistent_shape(sigdriv, sigdriv_imag)

        if self.use_driver_phase:
            phase, _ = phase_amplitude(sigdriv, phase=True, amplitude=False)
            sigdriv = np.cos(phase)
            if self.ordriv_d_ > 0:
                sigdriv_imag = np.sin(phase)

        test_mask = check_array(test_mask, dtype='bool', accept_none=True)

        if train_mask is not None:
            train_mask = check_array(train_mask, dtype='bool')
            check_consistent_shape(sigdriv, train_mask)

            # if test_mask is None, we use train_mask
            if test_mask is None:
                test_mask = train_mask
                both_masks = train_mask
            else:
                both_masks = np.logical_and(test_mask, train_mask)

            # we remove far masked data (by both masks)
            train_mask, test_mask, sigin, sigdriv, sigdriv_imag = \
                self.remove_far_masked_data(
                    both_masks,
                    [train_mask, test_mask, sigin, sigdriv, sigdriv_imag])

            check_consistent_shape(sigdriv, train_mask)

        if self.center:
            sigin -= np.mean(sigin)

        # -------- save signals as attributes of the model
        self.sigin = sigin
        self.sigdriv = sigdriv
        self.sigdriv_imag = sigdriv_imag
        self.train_mask = train_mask
        self.test_mask = test_mask

    def fit(self, sigin, sigdriv, fs, sigdriv_imag=None,
            train_mask=None, test_mask=None):
        """ Estimate a DAR model from input signals.

        Parameters
        ----------
        sigin : array, shape (n_epochs, n_points)
            Signal that is to be modeled

        sigdriv : array, shape (n_epochs, n_points)
            Signal that drives the model

        fs : float > 0
            Sampling frequency

        sigdriv_imag : None or array, shape (n_epochs, n_points)
            Second driver, containing the imaginary part of the driver

        train_mask : None or array, shape (n_epochs, n_points)
            If not None, the model is fitted only where 'train_mask' is False.

        test_mask : None or array, shape (n_epochs, n_points)
            If not None, the model is tested only where 'test_mask' is False.
            If 'test_mask' is None, it set equal to 'train_mask'.

        Returns
        -------
        self
        """
        self.reset_criterions()
        self.check_all_arrays(sigin, sigdriv, sigdriv_imag,
                              train_mask, test_mask)
        self.fs = fs

        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)

        # -------- estimation of the model
        if self.criterion:
            # -------- select the best order
            self.order_selection(self.criterion)
            self.estimate_error()
            self.estimate_gain()

        else:
            # -------- estimate a single model
            self.make_basis()
            self.estimate_ar()
            self.estimate_error()
            self.estimate_gain()
        return self

    def compute_cross_orders(self, ordriv, ordriv_d, cross_term_driver):
        max_order_cross_term = max(ordriv, ordriv_d)

        power_list, power_list_d = [], []

        # power of the driver
        for j in np.arange(1, ordriv + 1):
            power_list.append(j)
            power_list_d.append(0)

        # power of the derivative of the driver
        for k in np.arange(1, ordriv_d + 1):
            power_list.append(0)
            power_list_d.append(k)

        if cross_term_driver:
            # cross terms
            for j in range(1, ordriv):
                for k in range(1, ordriv_d):
                    if j + k >= max_order_cross_term:
                        power_list.append(j)
                        power_list_d.append(k)

        self.n_basis = len(power_list) + 1
        return power_list, power_list_d, self.n_basis

    def make_basis(self, sigdriv=None, sigdriv_imag=None):
        """Creates a basis from the driving signal, with the successive
        powers of sigdriv and sigdriv_imag.

        If self.normalize is True, each component of the basis will be
        normalized by its standard deviation.
        If self.ortho is True, a Gram-Schmidt orthogonalisation of the
        basis will be performed.

        During the fit, sigdriv and sigdriv_imag are not specified,
        it uses self.sigdriv and self.sigdriv_imag, and the basis is stored
        in self.basis_, along with the orthonormalization transform
        (self.alpha_). The method returns None.

        During the transform, sigdriv and sigdriv_imag are specified,
        the stored transform (self.alpha_) is applied on the new basis,
        and the method returns the new basis but does not store it.

        Parameters
        ----------
        sigdriv : None or array, shape (n_epochs, n_points)
            Signal that drives the model
            If None, we use self.sigdriv instead

        sigdriv_imag : None or array, shape (n_epochs, n_points)
            Second driver, containing the imaginary part of the driver

        Returns
        -------
        None or basis: array, shape (n_basis, n_epochs, n_points)
            Basis formed of the successive power of sigdriv and sigdriv_imag

        """
        # -------- check default arguments
        if sigdriv is None:
            sigdriv = self.sigdriv
            sigdriv_imag = self.sigdriv_imag
            save_basis = True
            ortho, normalize = self.ortho, self.normalize
        else:
            # reuse previous transformation
            sigdriv = check_array(sigdriv)
            sigdriv_imag = check_array(sigdriv_imag, accept_none=True)
            save_basis = False
            ortho, normalize = False, False

        n_epochs, n_points = sigdriv.shape
        ordriv_, ordriv_d_ = self.ordriv_, self.ordriv_d_
        power_list, power_list_d, n_basis = self.compute_cross_orders(
            ordriv_, ordriv_d_, self.cross_term_driver)

        # -------- memorize in alpha the various transforms
        alpha = np.eye(n_basis)

        # -------- create the basis
        basis = np.zeros((n_basis, n_epochs * n_points))
        basis[0] = 1.0

        # -------- create derivative if sigdriv_imag is not given
        if ordriv_d_ > 0:
            if sigdriv_imag is None:
                sigdriv_d = np.copy(sigdriv)
                sigdriv_d -= np.roll(sigdriv, shift=1, axis=1)
                sigdriv_d[:, 0] = sigdriv_d[:, 1]
                sigdriv_d *= sigdriv.std() / sigdriv_d.std()
            else:
                sigdriv_d = np.atleast_2d(sigdriv_imag)

        # -------- create successive components
        for k, (power, power_d) in enumerate(zip(power_list, power_list_d)):
            k += 1
            if power_d > 0 and power == 0:
                basis[k] = sigdriv_d.ravel() ** power_d
            elif power_d == 0 and power > 0:
                basis[k] = sigdriv.ravel() ** power
            elif power_d > 0 and power > 0:
                basis[k] = sigdriv_d.ravel() ** power_d
                basis[k] *= sigdriv.ravel() ** power
            else:
                raise ValueError('impossible !')

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
        # (basis = np.dot(alpha, rawbasis))
        if save_basis:
            self.basis_ = basis.reshape(-1, n_epochs, n_points)
            self.alpha_ = alpha
        else:
            if self.normalize or self.ortho:
                basis = np.dot(self.alpha_, basis)
            return basis.reshape(-1, n_epochs, n_points)

    def estimate_ar(self):
        """Estimates the AR model on a signal

        uses self.sigin, self.sigdriv and self.basis_ (which must be
        computed with make_basis before calling estimate_ar)
        self.ordar defines the order
        self.ordar_ receives this value after estimation
        """
        # -------- verify all parameters for the estimator
        if self.ordar < 1:
            raise ValueError('self.ordar is zero or negative')
        if self.ordriv_ < 0:
            raise ValueError('self.ordriv is negative')
        if self.ordriv_d_ < 0:
            raise ValueError('self.ordriv_d is negative')
        if self.basis_ is None:
            raise ValueError('basis does not yet exist')

        # -------- compute the final estimate
        if self.progress_bar:
            bar = ProgressBar(title=self.get_title(name=True))
        for AR_ in self.last_model():
            self.AR_ = AR_
            self.ordar_ = AR_.shape[0]
            if self.progress_bar:
                bar.update(float(self.ordar_) / self.ordar,
                           title=self.get_title(name=True))

    def remove_far_masked_data(self, mask, list_signals):
        """Remove unnecessary data which is masked
        and far (> self.ordar) from the unmasked data.
        """
        if mask is None:
            return list_signals

        selection = ~mask

        # convolution with a delay kernel,
        # so we keep the close points before the selection
        kernel = np.ones(self.ordar * 2 + 1)
        kernel[-self.ordar:] = 0.
        delayed_selection = fftconvolve(selection, kernel[None, :],
                                        mode='same')
        # remove numerical error from fftconvolve
        delayed_selection[np.abs(delayed_selection) < 1e-13] = 0.

        time_selection = delayed_selection.sum(axis=0) != 0
        epoch_selection = delayed_selection.sum(axis=1) != 0

        if not np.any(time_selection) or not np.any(epoch_selection):
            raise ValueError("The mask seems to hide everything.")

        output_signals = []
        for sig in list_signals:
            if sig is not None:
                sig = sig[..., epoch_selection, :]
                sig = sig[..., :, time_selection]
            output_signals.append(sig)

        return output_signals

    def get_train_data(self, sig):
        train_mask = self.train_mask
        train_mask, sig = self.remove_far_masked_data(
            train_mask, [train_mask, sig])

        return train_mask, sig

    def get_test_data(self, sig):
        test_mask = self.test_mask
        test_mask, sig = self.remove_far_masked_data(
            test_mask, [test_mask, sig])

        return test_mask, sig

    def degrees_of_freedom(self):
        return ((self.ordar_ + 1) * self.n_basis)

    def get_title(self, name=False, criterion=None):
        title = ''
        if name:
            title += self.__class__.__name__

        ordar_ = self.ordar_
        ordriv_ = self.ordriv_
        ordriv_d_ = self.ordriv_d_
        if ordriv_d_ > 0:
            title += '(%d, %d+%d)' % (ordar_, ordriv_, ordriv_d_)
        else:
            title += '(%d, %d)' % (ordar_, ordriv_)

        if criterion is not None:
            title += '_%s=%.4f' % (criterion, self.get_criterion(criterion))

        return title

    def get_criterion(self, criterion):
        criterion = criterion.lower()
        try:
            value = self.compute_criterions()[criterion]
        except:
            raise KeyError('Wrong criterion: %s' % criterion)
        return value

    def compute_criterions(self):
        criterions = getattr(self, 'criterions_', None)
        if criterions is not None:
            return criterions

        # else compute the criterions base on the log likelihood
        logl, tmax = self.log_likelihood(skip=self.ordar_)
        degrees = self.degrees_of_freedom()

        # Akaike Information Criterion
        eta_aic = 2.0
        aic = -2.0 * logl + eta_aic * degrees

        # Bayesian Information Criterion
        eta_bic = np.log(tmax)
        bic = -2.0 * logl + eta_bic * degrees

        self.criterions_ = {'aic': aic / tmax, 'bic': bic / tmax,
                            'logl': logl / tmax, '-logl': -logl / tmax,
                            'tmax': tmax}

        return self.criterions_

    @property
    def logl(self):
        return self.compute_criterions()['logl']

    @property
    def aic(self):
        return self.compute_criterions()['aic']

    @property
    def bic(self):
        return self.compute_criterions()['bic']

    def reset_criterions(self):
        self.model_selection_criterions_ = None
        self.criterions_ = None

    def order_selection(self, criterion):
        """Select the order of the model with a criterion based on the
        log likelihood.

        Attributes ordar and ordriv define the maximum values.
        This function saves the optimal values as ordar_ and ordriv_
        """
        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)
        ordar = self.ordar
        ordriv = self.ordriv
        ordriv_d = self.ordriv_d

        best_criterion = {'aic': np.inf, 'bic': np.inf, '-logl': np.inf}

        logl = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))
        aic = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))
        bic = np.empty((ordar + 1, ordriv + 1, ordriv_d + 1))

        # backward compatibility
        if not isinstance(criterion, str) and criterion:
            criterion = 'bic'
        if criterion.lower() == 'logl':
            criterion = '-logl'

        # -------- loop on ordriv with a copy of the estimator
        if self.progress_bar:
            bar = ProgressBar(title='%s' % self.__class__.__name__,
                              max_value=logl.size)
            bar_k = 0
        for ordriv_d_ in range(ordriv_d + 1):
            for ordriv_ in range(ordriv + 1):
                A = self.copy()
                # A.ordriv = ordriv_
                A.ordriv_ = ordriv_
                # A.ordriv_d = ordriv_d_
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

                    # compute the residual on training data to fit the gain
                    A.estimate_error(train=True)
                    A.estimate_gain()

                    # compute the residual on testing data to compute
                    # the likelihood
                    A.estimate_error(train=False)
                    A.reset_criterions()
                    this_criterion = A.compute_criterions()
                    logl[ordar_, ordriv_, ordriv_d_] = this_criterion['logl']
                    aic[ordar_, ordriv_, ordriv_d_] = this_criterion['aic']
                    bic[ordar_, ordriv_, ordriv_d_] = this_criterion['bic']

                    # -------- actualize the best model
                    if this_criterion[criterion] < best_criterion[criterion]:
                        best_criterion = this_criterion
                        self.ordar_ = ordar_
                        self.ordriv_ = ordriv_
                        self.ordriv_d_ = ordriv_d_
                        self.AR_ = np.copy(AR_)
                        self.G_ = np.copy(A.G_)

        # store all criterions
        self.model_selection_criterions_ = \
            {'logl': logl, 'aic': aic, 'bic': bic, '-logl': -logl,
             'tmax': best_criterion['tmax']}
        self.criterions_ = best_criterion

        # recompute the basis with ordriv_ and ordriv_d_
        self.make_basis()

    def estimate_gain(self, regul=0.01):
        """Estimate the gain expressed over the basis by maximizing
        the likelihood, through a Newton-Raphson procedure

        The residual e(t) is modelled as a white noise with standard
        deviation: std(e(t)) = exp(b(0) + b(1)u(t) + ... + b(m)u(t)^M)

        iter_gain : maximum number of iterations
        eps_gain   : stop iteration when corrections get smaller than eps_gain
        regul     : regularization factor (for inversion of the Hessian)
        """
        # -------- estimate the gain (self.G_) given the following data
        # self.ordriv_   : order of the regression of the parameters
        # self.residual_ : residual signal (from ordar to n_points)
        # self.basis_    : basis of functions (build from sigdriv)

        # -------- compute these intermediate quantities
        # resreg    : regression residual (residual * basis)
        # residual  : copy of self.residual_
        # residual2 : squared residual

        iter_gain = self.iter_gain
        eps_gain = self.eps_gain
        ordar = self.ordar_

        # -------- get the training data
        train_mask, basis = self.get_train_data(self.basis_)
        train_selection = ~train_mask if train_mask is not None else None

        residual = self.residual_

        # -------- crop the ordar first values
        residual = residual[:, ordar:]
        basis = basis[:, :, ordar:]
        if train_selection is not None:
            train_selection = train_selection[:, ordar:]

        # concatenate the epochs since it does not change the computations
        # as in estimating the AR coefficients
        if train_selection is not None:
            residual = residual[train_selection]
            basis = basis[:, train_selection]
            train_selection = train_selection[train_selection].reshape(1, -1)
        residual = residual.reshape(1, -1)
        basis = basis.reshape(basis.shape[0], -1)

        residual2 = residual ** 2
        n_points = residual.size
        resreg = basis * residual
        self.G_ = np.zeros((1, self.n_basis))

        # -------- prepare an initial estimation
        # chose N = 3 * ordriv_ classes, estimate a standard deviation
        # on each class, and make a regression on the values of
        # the indexes for their centroid.
        # For a time-dependent signal, a good idea was to take
        # successive slices of the signal, but for signal driven
        # models, this would be wrong! Use levels of the driving
        # function instead.
        if self.n_basis > 1:
            index = np.argsort(basis[1, :])  # indexes to sort the basis
        else:
            index = np.arange(n_points, dtype=int)

        nbslices = 3 * self.n_basis        # number of slices
        lenslice = n_points // nbslices    # length of a slice
        e = np.zeros(nbslices)             # log energies

        # -------- prepare least-squares equations
        tmp = lenslice * np.arange(nbslices + 1)
        kmin = tmp[:-1]
        kmax = tmp[1:]
        kmid = (kmin + kmax) // 2
        for k, (this_min, this_max) in enumerate(zip(kmin, kmax)):
            e[k] = np.mean(residual2[0, index[this_min:this_max]])

        if train_selection is not None:
            masked_basis = basis * train_selection
        else:
            masked_basis = basis

        e = 0.5 * np.log(e)
        R = np.dot(basis[:, index[kmid]],
                   masked_basis[:, index[kmid]].T)
        r = np.dot(e, masked_basis[:, index[kmid]].T)

        # -------- regularize matrix R
        v = np.resize(linalg.eigvalsh(R), (1, self.n_basis))
        correction = regul * np.max(v)

        # add tiny correction on diagonal without a copy
        R.flat[::len(R) + 1] += correction

        # -------- compute regularized solution
        self.G_ = linalg.solve(R, r)
        self.G_.shape = (1, self.n_basis)

        # -------- prepare the likelihood (and its initial value)
        loglike = np.zeros(iter_gain + 1)
        loglike.fill(np.nan)

        # -------- refine this model (iteratively maximise the likelihood)
        for itnum in range(iter_gain):
            logsigma = np.dot(self.G_, basis)
            sigma2 = np.exp(2 * logsigma)

            train_mask = (~train_selection if train_selection is not None
                          else None)
            loglike[itnum] = wgn_log_likelihood(residual, sigma2, train_mask)

            gradient = np.sum(basis * (residual2 / sigma2 - 1.0), 1)
            hessian = -2.0 * np.dot(resreg / sigma2, resreg.T)
            dG = linalg.solve(hessian, gradient)
            self.G_ -= dG.T
            if np.amax(np.absolute(dG)) < eps_gain:
                iter_gain = itnum + 1
                break
        self.residual_bis_ = residual / np.sqrt(sigma2)

    def fit_transform(self, sigin, sigdriv, fs, sigdriv_imag=None, mask=None):
        """Same as fit, but return the residual instead of the model object
        """
        self.fit(sigin=sigin, sigdriv=sigdriv, fs=fs,
                 sigdriv_imag=sigdriv_imag, mask=mask)
        return self.residual_

    def transform(self, sigin, sigdriv, fs, sigdriv_imag=None, mask=None):
        """Whiten a signal with the already fitted model
        """
        self.reset_criterions()
        self.check_all_arrays(sigin, sigdriv, sigdriv_imag, mask)
        self.basis_ = self.make_basis(sigdriv=sigdriv,
                                      sigdriv_imag=sigdriv_imag)

        self.estimate_error(train=True, recompute=True)
        return self.residual_

    def develop_gain(self, basis, sigdriv=None, squared=False, log=False):
        # n_basis, n_epochs, n_points = basis.shape
        # 1, n_basis = self.G_.shape
        # n_epochs, n_points = gain.shape
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
        # -------- get the left-out data
        test_mask, basis = self.get_test_data(self.basis_)

        # skip first samples
        if test_mask is not None:
            test_mask = test_mask[:, skip:]
        residual = self.residual_[:, skip:]

        # -------- estimate the gain
        gain2 = self.develop_gain(basis[:, :, skip:], squared=True)

        # -------- compute the log likelihood from the residual
        logL = wgn_log_likelihood(residual, gain2, test_mask)

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
        # -------- verify that the models have been fitted
        if not hasattr(self, 'ordar_'):
            raise ValueError('model was not fit before LR test.')
        if not hasattr(ar0, 'ordar_'):
            raise ValueError('ar0 was not fit before LR test.')

        # -------- compute the log likelihood for the models
        logL1, _ = self.log_likelihood(skip=self.ordar_)
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
    def estimate_error(self, train=True, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_
        """
        pass

    @abstractmethod
    def develop(self, basis, sigdriv):
        """Compute the AR models and gains at instants fixed by newcols

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

    def develop_all(self, sigdriv=None, sigdriv_imag=None):
        """
        Compute the AR models and gains for every instant of sigdriv
        """
        if sigdriv is None:
            sigdriv = self.sigdriv
            try:
                basis = self.basis_
            except:
                basis = self.make_basis()
        else:
            basis = self.make_basis(sigdriv=sigdriv, sigdriv_imag=sigdriv_imag)

        AR_cols, G_cols = self.develop(basis=basis, sigdriv=sigdriv)

        return AR_cols, G_cols, np.arange(sigdriv.size), sigdriv

    # ------------------------------------------------ #
    # Functions to plot the models                     #
    # ------------------------------------------------ #
    def _basis2spec(self, sigdriv, sigdriv_imag=None, frange=None):
        """Compute the power spectral density for a given basis
        frange  : frequency range

        this method is not intended for general use, except as a
        factorisation of the code of methods plot_time_freq and
        plot_drive_freq
        """
        # -------- distinguish estimated or target model
        ordar = self.ordar_

        # -------- Compute the AR models and gains for every instant of sigdriv
        if sigdriv is None:
            sigdriv = self.sigdriv
            basis = self.basis_
        else:
            basis = self.make_basis(sigdriv=sigdriv, sigdriv_imag=sigdriv_imag)

        AR_cols, G_cols = self.develop(basis=basis, sigdriv=sigdriv)

        # keep the only epoch
        AR_cols = AR_cols[:, 0, :]
        G_cols = G_cols[:1, :]

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
                            xlim=None):
        """Computes an amplitude-frequency power spectral density

        nbcols : number of expected columns (amplitude)
        frange : frequency range
        mode   : normalisation mode ('c' = centered, 'v' = unit variance)
        xlim   : minimum and maximum amplitude

        returns:
        spec : ndarray containing the time-frequency psd
        xlim : minimum and maximum amplitude

        """
        xlim, sigdriv, sigdriv_imag = self._driver_range(nbcols, xlim)

        # -------- compute spectra
        spec = self._basis2spec(sigdriv[None, :], sigdriv_imag=sigdriv_imag,
                                frange=frange)

        # -------- normalize
        if 'c' in mode:
            spec -= np.mean(spec, axis=1)
        if 'v' in mode:
            spec /= np.std(spec, axis=1)

        return spec, xlim, sigdriv, sigdriv_imag

    def _driver_range(self, nbcols=256, xlim=None):
        # -------- define the horizontal axis
        if xlim is None:
            xlim = self.get_sigdriv_bounds()

        if self.ordriv_d > 0 or False:
            # full oscillation for derivative
            phase = np.linspace(0, 2 * np.pi, nbcols, endpoint=False)
            sigdriv = xlim[1] * np.cos(phase)
            sigdriv_imag = xlim[1] * np.sin(phase)
        else:
            # simple ramp
            eps = 0.0
            sigdriv = np.linspace(xlim[0] + eps, xlim[1] - eps, nbcols)
            sigdriv_imag = None

        return xlim, sigdriv, sigdriv_imag

    def get_sigdriv_bounds(self, sigdriv=None):
        if self.use_driver_phase:
            bounds = [-1, 1]

        else:
            if sigdriv is None:
                sigdriv = self.sigdriv
            if self.train_mask is not None:
                sigdriv = sigdriv[~self.train_mask]
            bounds = np.percentile(sigdriv, [5, 95])
        bound_min = min(-bounds[0], bounds[1])
        bounds = (-bound_min, bound_min)
        return bounds

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
        return self.get_title(name=True)

    def __str__(self):
        return self.get_title(name=True)


def wgn_log_likelihood(eps, sigma2, mask=None):
    """Returns the likelihood of a gaussian uncorrelated signal
    given a model of its variance

    eps    : gaussian signal (residual or innovation)
    sigma2 : variance of this signal
    """
    if mask is not None:
        eps = eps[~mask]
        sigma2 = sigma2[~mask]

    tmax = eps.size
    logL = tmax * np.log(2.0 * np.pi)
    logL += np.sum(eps * eps / sigma2)
    logL += np.sum(np.log(sigma2))
    logL *= -0.5

    return logL


def check_array(sig, dtype='float64', accept_none=False):
    if accept_none and sig is None:
        return None
    elif not accept_none and sig is None:
        raise ValueError("This array should not be None.")

    return np.atleast_2d(np.array(sig, dtype=dtype))


def check_consistent_shape(*args):
    for array in args:
        if array.shape != args[0].shape:
            raise ValueError('Incompatible shapes. Got '
                             '(%s != %s)' % (array.shape, args[0].shape))
