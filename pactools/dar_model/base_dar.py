from abc import ABCMeta, abstractmethod
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.signal import fftconvolve
from scipy import stats

from ..utils.progress_bar import ProgressBar
from ..utils.maths import squared_norm
# from ..utils.deprecation import ignore_warnings
from ..utils.validation import check_array, check_consistent_shape
from ..utils.validation import check_is_fitted
from ..utils.spectrum import Spectrum
from ..utils.viz import add_colorbar, compute_vmin_vmax, phase_string

EPSILON = np.finfo(np.float32).eps


class BaseDAR(object):
    __metaclass__ = ABCMeta

    def __init__(self, ordar=1, ordriv=0, criterion=None, normalize=True,
                 ortho=True, center=True, iter_gain=10, eps_gain=1.0e-4,
                 progress_bar=False, use_driver_phase=False, max_ordar=None,
                 warn_gain_estimation_failure=False):
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
        self.warn_gain_estimation_failure = warn_gain_estimation_failure

        # for fair loglikelihood comparison
        self.max_ordar = max_ordar

        # power of the Taylor expansion
        self.ordriv = ordriv
        self._compute_cross_orders(ordriv)

        # weighting system
        self.train_weights = None
        self.test_weights = None

        # prepare other arrays
        self.basis_ = None

    def _check_all_arrays(self, sigin, sigdriv, sigdriv_imag, train_weights,
                          test_weights):
        # -------- transform the signals to 2d array of float64
        sigin = check_array(sigin)
        sigdriv = check_array(sigdriv)
        sigdriv_imag = check_array(sigdriv_imag, accept_none=True)
        check_consistent_shape(sigin, sigdriv, sigdriv_imag)
        check_consistent_shape(sigdriv, sigdriv_imag)

        train_weights = check_array(train_weights, accept_none=True)
        test_weights = check_array(test_weights, accept_none=True)

        if train_weights is not None:
            self.train_mask_ = train_weights == 0
            check_consistent_shape(sigdriv, train_weights)

            # if test_weights is None, we use test_weights = train_weights
            if test_weights is None:
                mask = train_weights == 0
            else:
                mask = np.logical_and(train_weights == 0, test_weights == 0)

            # we remove far masked data (by both masks)
            train_weights, test_weights, sigin, sigdriv, sigdriv_imag = \
                self._remove_far_masked_data(
                    mask, [train_weights, test_weights, sigin,
                           sigdriv, sigdriv_imag])

            self.train_mask_ = train_weights == 0
            if test_weights is not None:
                self.test_mask_ = test_weights == 0
            else:
                self.test_mask_ = None

            check_consistent_shape(sigdriv, train_weights)
        else:
            self.train_mask_ = None
            self.test_mask_ = None

        if self.use_driver_phase and self.ordriv > 0:
            if sigdriv_imag is None:
                raise ValueError('Impossible to use use_driver_phase=True '
                                 'without giving sigdriv_imag.')
            amplitude = np.sqrt(sigdriv ** 2 + sigdriv_imag ** 2)
            sigdriv = sigdriv / amplitude
            sigdriv_imag = sigdriv_imag / amplitude

        if self.center:
            sigin = sigin - np.mean(sigin)

        # -------- save signals as attributes of the model
        self.sigin = sigin
        self.sigdriv = sigdriv
        self.sigdriv_imag = sigdriv_imag
        self.train_weights = train_weights
        self.test_weights = test_weights

    def fit(self, sigin, sigdriv, fs, sigdriv_imag=None, train_weights=None,
            test_weights=None):
        """Estimate the model from input signals

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

        train_weights : None or array, shape (n_epochs, n_points)
            If not None, the model is fitted with these sample weights.

        test_weights : None or array, shape (n_epochs, n_points)
            If not None, the model is tested with these sample weights.
            If 'test_weights' is None, it is set equal to 'train_weights'.

        Returns
        -------
        self
        """
        self._reset_criterions()
        self._check_all_arrays(sigin, sigdriv, sigdriv_imag, train_weights,
                               test_weights)
        self.fs = fs

        # -------- check parameters
        if self.ordar < 0:
            raise ValueError('self.ordar is negative')
        if self.ordriv < 0:
            raise ValueError('self.ordriv is negative')

        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)

        # -------- estimation of the model
        if self.criterion:
            # -------- select the best order
            self._order_selection()
            self._estimate_error(recompute=True)

        else:
            self._fit()

        self._estimate_fit_std()

        return self

    def _fit(self):
        # -------- estimate a single model
        self.ordriv_ = self.ordriv
        self._make_basis()
        self._estimate_ar()
        self._estimate_error(recompute=self.test_weights is not None)
        self._estimate_gain()

    def _compute_cross_orders(self, ordriv):
        power_list_re, power_list_im = [], []

        # power of the driver
        for j in np.arange(ordriv + 1):
            for k in np.arange(j + 1):
                power_list_re.append(j - k)
                power_list_im.append(k)

        self.n_basis = len(power_list_re)
        return power_list_re, power_list_im, self.n_basis

    def _make_basis(self, sigdriv=None, sigdriv_imag=None, ordriv=None):
        """Creates a basis from the driving signal

        Creates a basis with the successive powers of sigdriv and sigdriv_imag.

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
        if ordriv is None:
            ordriv = self.ordriv
        if sigdriv_imag is None:
            power_list_re = np.arange(ordriv + 1)
            power_list_im = np.zeros(ordriv + 1)
            n_basis = ordriv + 1
            self.n_basis = ordriv + 1
        else:
            power_list_re, power_list_im, n_basis = self._compute_cross_orders(
                ordriv)

        # -------- memorize in alpha the various transforms
        alpha = np.eye(n_basis)

        # -------- create the basis
        basis = np.zeros((n_basis, n_epochs * n_points))

        # -------- create successive components
        for k, (power_real,
                power_imag) in enumerate(zip(power_list_re, power_list_im)):
            if power_imag == 0 and power_real == 0:
                basis[k] = 1.0
            elif power_imag > 0 and power_real == 0:
                basis[k] = sigdriv_imag.ravel() ** power_imag
            elif power_imag == 0 and power_real > 0:
                basis[k] = sigdriv.ravel() ** power_real
            elif power_imag > 0 and power_real > 0:
                basis[k] = sigdriv_imag.ravel() ** power_imag
                basis[k] *= sigdriv.ravel() ** power_real
            else:
                raise ValueError('Power cannot be negative : (%s, %s)' %
                                 (power_real, power_imag))

            if ortho:
                # correct current component with corrected components
                for m in range(k):
                    alpha[k, m] = -(np.sum(basis[k] * basis[m]) / np.sum(
                        basis[m] * basis[m]))
                basis[k] += np.dot(alpha[k, :k], basis[:k, :])
                # recompute the expression over initial components
                alpha[k, :k] = np.dot(alpha[k, :k], alpha[:k, :k])
            if normalize:
                scale = np.sqrt(
                    float(n_epochs * n_points) / squared_norm(basis[k]))
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

    def _estimate_ar(self):
        """Estimates the AR model on a signal

        uses self.sigin, self.sigdriv and self.basis_ (which must be
        computed with _make_basis before calling _estimate_ar)
        self.ordar defines the order
        """
        # -------- compute the final estimate
        if self.progress_bar:
            bar = ProgressBar(title=self.get_title(name=True))
        for AR_ in self._last_model():
            self.AR_ = AR_
            if self.progress_bar:
                bar.update(
                    float(self.ordar_) / self.ordar, title=self.get_title(
                        name=True))

    def _remove_far_masked_data(self, mask, list_signals):
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

    def _get_train_data(self, sig_list):
        if not isinstance(sig_list, list):
            sig_list = list(sig_list)

        train_weights = self.train_weights
        sig_list.append(train_weights)
        sig_list = self._remove_far_masked_data(self.train_mask_, sig_list)

        return sig_list

    def _get_test_data(self, sig_list):
        if not isinstance(sig_list, list):
            sig_list = list(sig_list)

        test_weights = self.test_weights
        if test_weights is None:
            test_weights = self.train_weights
        sig_list.append(test_weights)
        sig_list = self._remove_far_masked_data(self.test_mask_, sig_list)
        return sig_list

    def degrees_of_freedom(self):
        """Number of parameters of the fitted model"""
        return ((self.ordar_ + 1) * self.n_basis)

    def get_title(self, name=True, criterion=None):
        """Get the name and orders of the model"""
        title = ''
        if name:
            title += self.__class__.__name__

        if hasattr(self, 'AR_'):
            ordar_ = self.ordar_
            ordriv_ = self.ordriv_
            title += '(%d, %d)' % (ordar_, ordriv_)

            if criterion is not None and criterion is not False:
                title += '_%s=%.4f' % (criterion,
                                       self.get_criterion(criterion))
        else:
            # non fitted model
            ordar = self.ordar
            ordriv = self.ordriv
            title += '(%d, %d)' % (ordar, ordriv)

        return title

    def get_criterion(self, criterion, train=False):
        """Get the criterion (logL, AIC, BIC) of the fitted model

        If a self.criterion is not None, the fit performs a grid-search over
        the parameter ordar and ordriv. In such case, this method returns the
        array of value of the criterion over each parameter.
        """
        criterion = criterion.lower()
        try:
            value = self._compute_criterion(train=train)[criterion]
        except KeyError:
            raise KeyError('Wrong criterion: %s' % criterion)
        return value

    def _compute_criterion(self, train=False):
        check_is_fitted(self, 'AR_')
        criterions = getattr(self, 'criterions_', None)
        if criterions is not None:
            return criterions

        # else compute the criterions base on the log likelihood
        max_ordar = self.max_ordar
        if self.max_ordar is None:
            max_ordar = self.ordar
        logl, tmax = self._estimate_log_likelihood(train=train, skip=max_ordar)
        degrees = self.degrees_of_freedom()

        # Akaike Information Criterion
        eta_aic = 2.0
        aic = -2.0 * logl + eta_aic * degrees

        # Bayesian Information Criterion
        eta_bic = np.log(tmax)
        bic = -2.0 * logl + eta_bic * degrees

        self.criterions_ = {
            'aic': aic / tmax,
            'bic': bic / tmax,
            'logl': logl / tmax,
            '-logl': -logl / tmax,
            'tmax': tmax
        }

        return self.criterions_

    @property
    def logl(self):
        """Log likelihood of the model"""
        return self._compute_criterion()['logl']

    @property
    def aic(self):
        """Akaike information criterion (AIC) of the model"""
        return self._compute_criterion()['aic']

    @property
    def bic(self):
        """Bayesian information criterion (BIC) of the model"""
        return self._compute_criterion()['bic']

    @property
    def tmax(self):
        """Scaling of self.logl, self.aic, and self.bic"""
        return self._compute_criterion()['tmax']

    def _reset_criterions(self):
        """Reset stored criterions (negative log_likelihood, AIC or BIC)"""
        self.model_selection_criterions_ = None
        self.criterions_ = None

    def _order_selection(self):
        """Fit several models with a grid-search over self.ordar and
        self.ordriv, and select the model with the best criterion
        self.criterion (negative log_likelihood, AIC or BIC)
        """
        # compute the basis once and for all
        self._make_basis()

        criterion = self.criterion
        # -------- prepare the estimates
        self.AR_ = np.ndarray(0)
        self.G_ = np.ndarray(0)
        ordar = self.ordar
        ordriv = self.ordriv

        best_criterion = {'aic': np.inf, 'bic': np.inf, '-logl': np.inf}

        logl = np.zeros((ordar + 1, ordriv + 1))
        aic = np.zeros((ordar + 1, ordriv + 1))
        bic = np.zeros((ordar + 1, ordriv + 1))

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
        for ordriv in range(ordriv + 1):
            model = self.copy()
            model.ordriv = ordriv
            model.ordriv_ = ordriv
            _, _, n_basis = model._compute_cross_orders(ordriv)
            model.basis_ = self.basis_[:n_basis]

            # -------- estimate the best AR order for this value of ordriv
            for AR_ in model._next_model():
                model.AR_ = AR_
                if self.progress_bar:
                    title = model.get_title(name=True)
                    bar_k += 1
                    bar.update(bar_k, title=title)

                model._estimate_error(recompute=self.test_weights is not None)
                model._estimate_gain()
                model._reset_criterions()
                this_criterion = model._compute_criterion()
                logl[model.ordar_, model.ordriv_] = this_criterion['logl']
                aic[model.ordar_, model.ordriv_] = this_criterion['aic']
                bic[model.ordar_, model.ordriv_] = this_criterion['bic']

                # -------- actualize the best model
                if this_criterion[criterion] < best_criterion[criterion]:
                    best_criterion = this_criterion
                    self.AR_ = np.copy(model.AR_)
                    self.G_ = np.copy(model.G_)
                    self.ordriv_ = model.ordriv_

        # store all criterions
        self.model_selection_criterions_ = \
            {'logl': logl, 'aic': aic, 'bic': bic, '-logl': -logl,
             'tmax': best_criterion['tmax']}
        self.criterions_ = best_criterion

        # select the corresponding basis
        _, _, n_basis = self._compute_cross_orders(self.ordriv_)
        assert self.AR_.shape[1] == n_basis
        self.basis_ = self.basis_[:n_basis]
        self.alpha_ = self.alpha_[:n_basis, :n_basis]

    @property
    def ordar_(self):
        """AR order of the model, different from self.ordar if a model
        selection has been performed
        """
        check_is_fitted(self, 'AR_')
        return self.AR_.shape[0]

    def _estimate_fixed_gain(self):
        """Estimates a fixed gain from the predicton error self.residual_.
        """
        # -------- get the training data
        residual, weights = self._get_train_data([self.residual_])

        # -------- crop the ordar first values
        residual = residual[:, self.ordar_:]
        if weights is not None:
            weights = weights[:, self.ordar_:]

        # -------- apply the mask
        if weights is not None:
            average = np.average(residual, weights=weights)
            sigma2 = np.average((residual - average) ** 2, weights=weights)
            sigma = np.sqrt(sigma2)
        else:
            sigma = np.std(residual)

        # -------- estimate an initial (fixed) model of the residual
        self.G_ = np.zeros((1, self.n_basis))
        self.G_[0, 0] = np.log(sigma)

    def _estimate_gain(self, regul=0.):
        """helper to handle failure in gain estimation"""
        if self.ordriv == 0:
            self._estimate_fixed_gain()
            return None

        try:
            self._estimate_driven_gain(regul=regul)
        except Exception as e:
            if self.warn_gain_estimation_failure:
                msg = 'Gain estimation failed, fixed gain used. Error: %s' % e
                warnings.warn(msg)
            self._estimate_fixed_gain()

    def _estimate_driven_gain(self, regul=0.):
        """Estimate the gain expressed over the basis by maximizing
        the likelihood, through a Newton-Raphson procedure

        The residual e(t) is modelled as a white noise with standard
        deviation driven by the driver x:
            std(e(t)) = exp(b(0) + b(1)x(t) + ... + b(m)x(t)^m)

        iter_gain : maximum number of iterations
        eps_gain  : stop iteration when corrections get smaller than eps_gain
        regul     : regularization factor (for inversion of the Hessian)
        """
        iter_gain = self.iter_gain
        eps_gain = self.eps_gain
        ordar_ = self.ordar_

        # -------- get the training data
        basis, residual, weights = self._get_train_data(
            [self.basis_, self.residual_])

        residual += EPSILON

        # -------- crop the ordar first values
        residual = residual[:, ordar_:]
        basis = basis[:, :, ordar_:]
        if weights is not None:
            weights = weights[:, ordar_:]

        # concatenate the epochs since it does not change the computations
        # as in estimating the AR coefficients
        residual = residual.reshape(1, -1)
        basis = basis.reshape(basis.shape[0], -1)
        if weights is not None:
            weights = weights.reshape(1, -1)

        residual2 = residual ** 2
        n_points = residual.size
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

        nbslices = 3 * self.n_basis  # number of slices
        lenslice = n_points // nbslices  # length of a slice
        e = np.zeros(nbslices)  # log energies

        # -------- prepare least-squares equations
        tmp = lenslice * np.arange(nbslices + 1)
        kmin = tmp[:-1]
        kmax = tmp[1:]
        kmid = (kmin + kmax) // 2
        for k, (this_min, this_max) in enumerate(zip(kmin, kmax)):
            indices = index[this_min:this_max]
            if weights is not None:
                e[k] = np.average(residual2[0, indices],
                                  weights=weights[0, indices])
            else:
                e[k] = np.mean(residual2[0, indices])

        # this least-square is not weighted, only the construction of e
        e = 0.5 * np.log(e)
        R = np.dot(basis[:, index[kmid]], basis[:, index[kmid]].T)
        r = np.dot(e, basis[:, index[kmid]].T)

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

        if iter_gain > 0:
            resreg = basis * residual
            if weights is not None:
                weighted_basis = weights * basis
                weighted_resreg = weights * resreg
        # -------- refine this model (iteratively maximise the likelihood)
        for itnum in range(iter_gain):
            sigma2 = self._compute_sigma2(basis)
            loglike[itnum] = wgn_log_likelihood(residual, sigma2,
                                                weights=weights)

            tmp = residual2 / sigma2 - 1.0
            if weights is not None:
                gradient = np.sum(weighted_basis * tmp, 1)
                hessian = -2.0 * np.dot(weighted_resreg / sigma2, resreg.T)
            else:
                gradient = np.sum(basis * tmp, 1)
                hessian = -2.0 * np.dot(resreg / sigma2, resreg.T)
            dG = linalg.solve(hessian, gradient)
            self.G_ -= dG.T
            if np.amax(np.absolute(dG)) < eps_gain:
                iter_gain = itnum + 1
                break

        sigma2 = self._compute_sigma2(basis)
        self.residual_bis_ = residual / np.sqrt(sigma2)

    def _compute_sigma2(self, basis):
        """Helper to compute the instantaneous variance of the model"""
        # with ignore_warnings():
        logsigma = np.dot(self.G_, basis)
        sigma2 = np.exp(2 * logsigma) + EPSILON

        if sigma2.max() > 1e5:
            raise RuntimeError(
                '_estimate_gain: sigma2.max() = %f' % sigma2.max())
        return sigma2

    def fit_transform(self, sigin, sigdriv, fs, sigdriv_imag=None,
                      train_weights=None, test_weights=None):
        """Same as fit, but returns the residual instead of the model object"""
        self.fit(sigin=sigin, sigdriv=sigdriv, fs=fs,
                 sigdriv_imag=sigdriv_imag, train_weights=train_weights,
                 test_weights=test_weights)
        return self.residual_

    def transform(self, sigin, sigdriv, fs, sigdriv_imag=None,
                  test_weights=None):
        """Whiten a signal with the already fitted model"""
        check_is_fitted(self, 'AR_')
        assert fs == self.fs
        self._reset_criterions()
        self._check_all_arrays(sigin, sigdriv, sigdriv_imag, None,
                               test_weights)
        self.basis_ = self._make_basis(
            sigdriv=sigdriv, sigdriv_imag=sigdriv_imag, ordriv=self.ordriv_)

        self._estimate_error(recompute=True)
        return self.residual_

    def _develop_gain(self, basis, sigdriv=None, squared=False, log=False):
        # n_basis, n_epochs, n_points = basis.shape
        # 1, n_basis = self.G_.shape
        # n_epochs, n_points = gain.shape
        G_cols = np.tensordot(self.G_.ravel(), basis, axes=([0], [0]))
        if squared:
            G_cols *= 2
        if not log:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                G_cols = np.exp(G_cols) + EPSILON * 10
        return G_cols

    def _estimate_fit_std(self, train=False, skip=None):
        """Estimate the likelihood of a model with ordar=0, ordriv=0"""
        if train:
            sigin, weights = self._get_train_data([self.sigin])
        else:
            sigin, weights = self._get_test_data([self.sigin])

        if skip is None:
            skip = self.ordar_
        sigin = sigin[:, skip:]
        if weights is not None:
            weights = weights[:, skip:]

        if weights is not None:
            average = np.average(sigin, weights=weights)
            sigma2 = np.average((sigin - average) ** 2, weights=weights)
            sigma = np.sqrt(sigma2)
        else:
            sigma = np.std(sigin)

        self.fit_std_ = sigma

    def _estimate_log_likelihood_ref(self, train=False, skip=0):
        """Estimate the likelihood of a model with ordar=0, ordriv=0"""
        if train:
            sigin, weights = self._get_train_data([self.sigin])
        else:
            sigin, weights = self._get_test_data([self.sigin])

        gain2 = self.fit_std_ ** 2

        # skip first samples
        sigin = sigin[:, skip:]
        if weights is not None:
            weights = weights[:, skip:]

        logL = wgn_log_likelihood(sigin, gain2, weights=weights)
        tmax = sigin.size

        return logL, tmax

    def _estimate_log_likelihood(self, train=False, skip=0):
        """Approximate computation of the logarithm of the likelihood

        The computation uses self.residual_, self.basis_ and self.G_

        skip : how many initial samples to skip
        """
        check_is_fitted(self, 'residual_')
        if train:
            basis, residual, weights = self._get_train_data(
                [self.basis_, self.residual_])
        else:
            basis, residual, weights = self._get_test_data(
                [self.basis_, self.residual_])

        # skip first samples
        residual = residual[:, skip:]
        basis = basis[:, :, skip:]
        if weights is not None:
            weights = weights[:, skip:]

        # -------- estimate the gain
        gain2 = self._develop_gain(basis, squared=True)

        # -------- compute the log likelihood from the residual
        logL = wgn_log_likelihood(residual, gain2, weights=weights)

        tmax = gain2.size
        return logL, tmax

    def likelihood_ratio(self, ar0):
        """Computation of the likelihood ratio test

        The null hypothesis H0 is given by model ar0; the current
        model (self) is tested against this model ar0 which may be
        either an invariant AR model (ordriv=0) or a hybrid one
        (invariant AR part, with driven gain).

        Parameters
        ----------
        ar0 : DAR model instance (fitted)
            Model representing the null hypothesis

        Returns:
        --------
        p_value : float
            The p-value for the test (1-  F(chi2(ratio)))
        """
        # -------- verify that the models have been fitted
        ordar_1 = self.ordar_
        ordar_0 = ar0.ordar_

        # -------- compute the log likelihood for the models
        logL1, _ = self._estimate_log_likelihood(skip=ordar_1)
        logL0, _ = ar0._estimate_log_likelihood(skip=ordar_0)

        # -------- compute the log likelihood ratio
        test = 2.0 * (logL1 - logL0)

        # -------- compute degrees of freedom
        df1 = self.degrees_of_freedom()
        df0 = ar0.degrees_of_freedom()

        # -------- compute the p-value
        p_value = stats.chi2.sf(test, df1 - df0)

        return p_value

    # ------------------------------------------------ #
    # Functions that should be overloaded              #
    # ------------------------------------------------ #
    @abstractmethod
    def _next_model(self):
        """Compute the AR model at successive orders

        Acts as a generator that stores the result in self.AR_
        Creates the models with orders from 0 to self.ordar

        Typical usage:
        for AR_ in A._next_model():
            A.AR_ = AR_
        """
        pass

    @abstractmethod
    def _estimate_error(self, recompute=False):
        """Estimates the prediction error

        uses self.sigin, self.basis_ and AR_
        """
        pass

    @abstractmethod
    def _develop(self, basis):
        """Compute the AR models and gains at instants fixed by newcols

        returns:
        ARcols : array containing the AR parts
        Gcols  : array containing the gains

        The size of ARcols is (1+ordar, len(newcols))
        The size of Gcols is (1, len(newcols))
        """
        pass

    def _last_model(self):
        """overload if it's faster to get only last model"""
        return self._next_model()

    def _develop_all(self, sigdriv=None, sigdriv_imag=None):
        """
        Compute the AR models and gains for every instant of sigdriv
        """
        if sigdriv is None:
            sigdriv = self.sigdriv
            try:
                basis = self.basis_
            except AttributeError:
                basis = self._make_basis(ordriv=self.ordriv_)
        else:
            basis = self._make_basis(sigdriv=sigdriv,
                                     sigdriv_imag=sigdriv_imag,
                                     ordriv=self.ordriv_)

        AR_cols, G_cols = self._develop(basis=basis)

        return AR_cols, G_cols, np.arange(sigdriv.size), sigdriv

    # ------------------------------------------------ #
    # Functions to plot the models                     #
    # ------------------------------------------------ #
    def _basis2spec(self, sigdriv, sigdriv_imag=None, frange=None, n_fft=256):
        """Compute the power spectral density for a given basis
        frange  : frequency range

        this method is not intended for general use, except as a
        factorisation of the code of methods plot_time_freq and
        plot_drive_freq
        """
        check_is_fitted(self, 'AR_')
        ordar_ = self.ordar_

        AR_cols, G_cols, _, sigdriv = self._develop_all(sigdriv, sigdriv_imag)
        # keep the only epoch
        AR_cols = AR_cols[:, 0, :]
        G_cols = G_cols[:1, :]

        # -------- estimate AR spectrum
        while n_fft < (ordar_ + 1):
            n_fft *= 2

        if ordar_ > 0:
            AR_cols = AR_cols / G_cols
            AR_spec = np.fft.rfft(AR_cols, n=n_fft, axis=0)
        else:
            Ginv_cols = 1.0 / G_cols
            AR_spec = np.fft.rfft(Ginv_cols, n=n_fft, axis=0)

        # -------- estimate AR spectrum
        spec = -20.0 * np.log10(np.abs(AR_spec))

        # -------- truncate to frange
        if frange is not None:
            frequencies = np.linspace(0, self.fs // 2, 1 + n_fft // 2)
            mask = np.logical_and(frequencies <= frange[1],
                                  frequencies >= frange[0])
            spec = spec[mask, :]
        return spec

    def _amplitude_frequency(self, nbcols=256, frange=None, mode='', xlim=None,
                             n_fft=256):
        """Computes an amplitude-frequency power spectral density

        nbcols : number of expected columns (amplitude)
        frange : frequency range
        mode   : normalisation mode ('c' = centered, 'v' = unit variance)
        xlim   : minimum and maximum amplitude

        returns:
        spec : ndarray containing the time-frequency psd
        xlim : minimum and maximum amplitude

        """
        check_is_fitted(self, 'AR_')
        xlim, sigdriv, sigdriv_imag = self._driver_range(nbcols, xlim)
        sigdriv = check_array(sigdriv)
        sigdriv_imag = check_array(sigdriv_imag, accept_none=True)

        # -------- compute spectra
        spec = self._basis2spec(sigdriv=sigdriv, sigdriv_imag=sigdriv_imag,
                                frange=frange, n_fft=n_fft)

        # -------- normalize
        if 'c' in mode:
            spec -= np.mean(spec, axis=1)[:, None]
        if 'v' in mode:
            spec /= np.std(spec, axis=1)[:, None]

        return spec, xlim, sigdriv, sigdriv_imag

    def _driver_range(self, nbcols=256, xlim=None):
        # -------- define the horizontal axis
        if xlim is None:
            xlim = self._get_sigdriv_bounds()

        # full oscillation for derivative
        phase = np.linspace(-np.pi, np.pi, nbcols, endpoint=False)
        if self.sigdriv_imag is None:
            sigdriv = np.linspace(xlim[0], xlim[1], nbcols)
            sigdriv_imag = None
        else:
            sigdriv = xlim[0] * np.cos(phase)
            sigdriv_imag = xlim[1] * np.sin(phase)

        return xlim, sigdriv, sigdriv_imag

    def _get_sigdriv_bounds(self, sigdriv=None, sigdriv_imag=None):
        if self.use_driver_phase:
            return [-1, 1]

        if sigdriv is None:
            sigdriv = self.sigdriv
        if sigdriv_imag is None:
            sigdriv_imag = self.sigdriv_imag

        if self.train_weights is not None:
            sigdriv = sigdriv[self.train_weights != 0]
            if sigdriv_imag is not None:
                sigdriv_imag = sigdriv_imag[self.train_weights != 0]

        if sigdriv_imag is None:
            bounds = np.percentile(sigdriv, [5, 95])
            # bound_min = min(-bounds[0], bounds[1])
            # bounds = (-bound_min, bound_min)
        else:
            bounds_real = np.sqrt(np.median(sigdriv ** 2))
            bounds_imag = np.sqrt(np.median(sigdriv_imag ** 2))
            bounds = (bounds_real, bounds_imag)

        return bounds

    def plot(self, title=None, frange=None, mode='c', vmin=None, vmax=None,
             ax=None, xlim=None, cmap=None, colorbar=True):
        """Plot the PSD as a function of the driver

        self  : the AR model to plot
        title : title of the plot
        frange: frequency range
        mode  : normalisation mode ('c' = centered, 'v' = unit variance)
        vmin  : minimum power to plot (dB)
        vmax  : maximum power to plot (dB)
        tick  : tick in the scale (dB)
        ax    : matplotlib.axes.Axes
        xlim  : force xlim to these values if defined
        """
        check_is_fitted(self, 'AR_')
        spec, xlim, sigdriv, sigdriv_imag = self._amplitude_frequency(
            mode=mode, xlim=xlim, frange=frange)
        self.spec_ = spec

        if cmap is None:
            if 'c' in mode:
                cmap = plt.get_cmap('RdBu_r')
            else:
                cmap = plt.get_cmap('inferno')

        if frange is None:
            frange = [0, self.fs / 2.0]

        if ax is None:
            fig = plt.figure(title)
            ax = fig.gca()
        else:
            fig = ax.figure

            # clean colorbar and image
            if ax.images != []:
                ax.images[-1].colorbar.remove()
            ax.cla()

        if sigdriv_imag is not None:
            extent = (-np.pi, np.pi, frange[0], frange[-1])
        else:
            extent = (xlim[0], xlim[1], frange[0], frange[-1])

        # -------- plot spectrum
        vmin, vmax = compute_vmin_vmax(spec, vmin, vmax, tick=.1, percentile=0)
        if 'c' in mode:
            vmax = max(vmax, -vmin)
            vmin = -vmax
        cax = ax.imshow(spec, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                        origin='lower', extent=extent, interpolation='none')

        if colorbar:
            # fig.tight_layout()
            add_colorbar(fig, cax, vmin, vmax, unit='dB', ax=ax)

        if title is None:
            title = self.get_title(name=True, criterion=self.criterion)
        ax.set_title(title)
        ax.set_ylabel('Frequency (Hz)')

        if sigdriv_imag is not None:
            ax.set_xlabel(r"Driver's phase $\phi_x$")
            ticks = np.arange(-np.pi, np.pi + np.pi / 2, np.pi / 2)
            ax.set_xticks(ticks)
            ax.set_xticklabels([r'$%s$' % phase_string(d) for d in ticks])
        else:
            ax.set_xlabel(r"Driver's value $x$")

        return fig, cax, vmin, vmax

    def plot_lines(self, title='', frange=None, mode='', ax=None, xlim=None,
                   vmin=None, vmax=None):
        """Plot the PSD as a function of the driver

        self    : the AR model to plot
        title   : title of the plot
        frange  : frequency range
        mode    : normalisation mode ('c' = centered, 'v' = unit variance)
        ax      : matplotlib.axes.Axes
        xlim    : force xlim to these values if defined
        vmin    : minimum power to plot (dB)
        vmax    : maximum power to plot (dB)
        """
        check_is_fitted(self, 'AR_')
        spec, xlim, sigdriv, sigdriv_imag = self._amplitude_frequency(
            mode=mode, xlim=xlim, frange=frange)
        self.spec_ = spec

        if ax is None:
            fig = plt.figure(title)
            ax = fig.gca()
        else:
            fig = ax.figure

        if frange is None:
            frange = [0, self.fs / 2.0]

        # -------- plot spectrum
        n_frequency, n_amplitude = spec.shape
        if sigdriv_imag is None:
            n_lines = 5
            ploted_amplitudes = np.linspace(0, n_amplitude - 1, n_lines)
        else:
            n_lines = 4
            ploted_amplitudes = np.linspace(0, n_amplitude, n_lines + 1)[:-1]
        ploted_amplitudes = np.floor(ploted_amplitudes).astype(np.int)

        frequencies = np.linspace(frange[0], frange[-1], n_frequency)
        for i, i_amplitude in enumerate(ploted_amplitudes[::-1]):

            # build label
            if sigdriv_imag is not None:
                sig_complex = (sigdriv[..., i_amplitude] +
                               1j * sigdriv_imag[..., i_amplitude])

                label = r'$\phi_x=%s$' % phase_string(sig_complex)
            else:
                label = r'$x=%.2f$' % sigdriv[0, i_amplitude]

            ax.plot(frequencies, spec[:, i_amplitude], label=label)

        # plot simple periodogram
        if True:
            spect = Spectrum(block_length=128, fs=self.fs, wfunc=np.blackman)
            sigin = self.sigin
            if self.train_weights is not None:
                sigin = sigin[self.train_weights != 0]
            spect.periodogram(sigin)
            fft_length, _ = spect.check_params()
            n_frequency = fft_length // 2 + 1
            psd = spect.psd[0][0, 0:n_frequency]
            psd = 10.0 * np.log10(np.maximum(psd, 1.0e-12))
            psd -= psd.mean()
            if 'c' not in mode:
                psd += 20.0 * np.log10(sigin.std())
            frequencies = np.linspace(0, spect.fs / 2, n_frequency)
            ax.plot(frequencies, psd, '--k', label=r'$PSD$')

        if title == '':
            title = self.get_title(name=True, criterion=self.criterion)
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power (dB)')
        ax.legend(loc=0)
        vmin, vmax = compute_vmin_vmax(spec, vmin, vmax, 0.1, percentile=0)
        vmin_, vmax_ = compute_vmin_vmax(psd, None, None, 0.1, percentile=0)
        vmin = min(vmin, vmin_)
        vmax = max(vmax, vmax_)
        ax.set_ylim((vmin, vmax))
        ax.grid('on')

        return fig

    # ------------------------------------------------ #
    def _to_dict(self):
        """Convert the model attributes to dict"""

        data = self.__dict__
        for k in data.keys():
            if '__' in k or k[0] == '_':
                del (data[k])
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
        data = self._to_dict()
        for k in data.keys():
            if k not in ('AR_', 'G_') and k[0] != '_':
                setattr(A, k, data[k])
        return A

    def __repr__(self):
        return self.get_title(name=True)

    def __str__(self):
        return self.get_title(name=True)


def wgn_log_likelihood(eps, sigma2, weights=None):
    """Returns the log-likelihood of a white Gaussian noise (WGN)

    (given a model of its variance)

    Parameters
    ----------
    eps : array, shape (n_epochs, n_points)
        Residual of the fit (a.k.a. innovation)

    sigma2 : array, shape (n_epochs, n_points)
        Estimated variance of eps

    weights : array, shape (n_epochs, n_points), or None (default None)
        Weights to apply to the log-likelihood for each time points

    Returns
    -------
    logL : float
        Log-likelihood of the residual as a white Gaussian noise (WGN)
    """
    if weights is not None:
        weights_sum = weights.sum()
    else:
        weights_sum = eps.size

    logL = weights_sum * np.log(2.0 * np.pi)

    if weights is not None:
        logL += np.sum(weights * (eps * eps / sigma2 + np.log(sigma2)))
    else:
        logL += np.sum(eps * eps / sigma2 + np.log(sigma2))

    logL *= -0.5

    return logL
