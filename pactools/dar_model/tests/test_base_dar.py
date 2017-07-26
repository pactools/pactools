import numpy as np
import matplotlib.pyplot as plt

from pactools.utils.testing import assert_equal, assert_array_almost_equal
from pactools.utils.testing import assert_greater, assert_raises
from pactools.utils.testing import assert_array_not_almost_equal
from pactools.dar_model import DAR, AR, HAR, StableDAR
from pactools.simulate_pac import simulate_pac

ALL_MODELS = [DAR, AR, HAR, StableDAR]

# Parameters used for the simulated sigin in the test
low_fq_range = [1., 3., 5., 7.]
high_fq_range = [25., 50., 75.]
n_low = len(low_fq_range)
n_high = len(high_fq_range)
high_fq = high_fq_range[1]
low_fq = low_fq_range[1]
n_points = 1024
fs = 200.

_sigin = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                      low_fq_width=1., noise_level=0.3, random_state=0)
_sigdriv = simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq,
                        low_fq=low_fq, low_fq_width=1., noise_level=0.,
                        random_state=0, high_fq_amp=0, return_driver=True)
_sigdriv_imag = np.imag(_sigdriv)
_sigdriv = np.real(_sigdriv)
_noise = np.random.RandomState(0).randn(n_points)

_model_params = {'ordar': 10, 'ordriv': 2, 'criterion': False}


def fast_fitted_model(klass=DAR, model_params=_model_params, sigin=_sigin,
                      sigdriv=_sigdriv, sigdriv_imag=_sigdriv_imag, fs=fs,
                      train_weights=None, test_weights=None):
    model_params = model_params.copy()
    if klass == StableDAR and 'iter_newton' not in model_params:
        model_params['iter_newton'] = 10
    return klass(**model_params).fit(
        sigin=sigin, sigdriv=sigdriv, sigdriv_imag=sigdriv_imag, fs=fs,
        train_weights=train_weights, test_weights=test_weights)


def test_likelihood_ratio_noise():
    # Test that likelihood_ratio returns a positive p_value with noise
    sdar = fast_fitted_model(StableDAR, sigin=_noise)
    dar = fast_fitted_model(DAR, sigin=_noise)
    ar = fast_fitted_model(AR, sigin=_noise)
    har = fast_fitted_model(HAR, sigin=_noise)

    for h0 in [ar, har]:
        for h1 in [dar, sdar]:
            p_value = h1.likelihood_ratio(h0)
            assert_greater(p_value, 0)


def test_likelihood_ratio_pac():
    # Test that likelihood_ratio returns a near zero p_value with PAC
    sdar = fast_fitted_model(StableDAR, sigin=_sigin)
    dar = fast_fitted_model(DAR, sigin=_sigin)
    ar = fast_fitted_model(AR, sigin=_sigin)
    har = fast_fitted_model(HAR, sigin=_sigin)

    for h0 in [ar, har]:
        for h1 in [dar, sdar]:
            p_value = h1.likelihood_ratio(h0)
            assert_greater(1e-3, p_value)


def test_degrees_of_freedom():
    # test the number of fitted parameters in the model
    model_params = {'ordar': 5, 'ordriv': 2, 'criterion': False}
    dar = fast_fitted_model(model_params=model_params)
    assert_equal(dar.degrees_of_freedom(), 6 * 6)

    model_params = {'ordar': 5, 'ordriv': 0, 'criterion': False}
    dar = fast_fitted_model(model_params=model_params)
    assert_equal(dar.degrees_of_freedom(), 6 * 1)

    model_params = {'ordar': 1, 'ordriv': 2, 'criterion': False}
    dar = fast_fitted_model(model_params=model_params)
    assert_equal(dar.degrees_of_freedom(), 2 * 6)

    model_params = {'ordar': 10, 'ordriv': 2, 'criterion': False}
    ar = fast_fitted_model(AR, model_params=model_params)
    assert_equal(ar.degrees_of_freedom(), 10 + 1)

    model_params = {'ordar': 10, 'ordriv': 2, 'criterion': False}
    har = fast_fitted_model(HAR, model_params=model_params)
    assert_equal(har.degrees_of_freedom(), 10 + 6)

    model_params = {'ordar': 5, 'ordriv': 2, 'criterion': False}
    har = fast_fitted_model(model_params=model_params, sigdriv_imag=None)
    assert_equal(har.degrees_of_freedom(), 6 * 3)

    model_params = {'ordar': 10, 'ordriv': 2, 'criterion': False}
    har = fast_fitted_model(HAR, model_params=model_params, sigdriv_imag=None)
    assert_equal(har.degrees_of_freedom(), 10 + 3)


def dar_no_fit(ortho, normalize, sigdriv=_sigdriv, sigdriv_imag=_sigdriv_imag,
               **model_params):
    dar = DAR(ortho=ortho, normalize=normalize, **model_params)
    dar.sigin = _sigin[None, :]
    dar.sigdriv = sigdriv[None, :]
    dar.sigdriv_imag = sigdriv_imag[None, :]
    dar._make_basis()
    return dar


def test_make_basis_new_sigdriv():
    # Test that _make_basis works the same with a new sigdriv,
    # using stored orthonormalization transform.
    model_params = {'ordar': 5, 'ordriv': 2, 'criterion': False}
    for normalize in (True, False):
        for ortho in (True, False):
            for this_sigdriv, this_sigdriv_imag in ([_sigdriv, _sigdriv_imag],
                                                    [_noise, _noise[::-1]]):
                dar = dar_no_fit(
                    ortho=ortho, normalize=normalize, sigdriv=this_sigdriv,
                    sigdriv_imag=this_sigdriv_imag, **model_params)
                newbasis = dar._make_basis(sigdriv=this_sigdriv,
                                           sigdriv_imag=this_sigdriv_imag)
                assert_array_almost_equal(newbasis, dar.basis_)

                # Different result if we change a parameter
                dar_ortho = dar_no_fit(not ortho, normalize, **model_params)
                dar_norma = dar_no_fit(ortho, not normalize, **model_params)
                for dar2 in [dar_ortho, dar_norma]:
                    assert_raises(AssertionError, assert_array_almost_equal,
                                  dar.basis_, dar2.basis_)


def test_make_basis_ortho_normalize():
    # Test the effect of ortho and normalize parameters
    model_params = {'ordar': 5, 'ordriv': 2, 'criterion': False}

    for normalize in (True, False):
        for ortho in (True, False):
            dar = dar_no_fit(ortho, normalize, **model_params)
            basis = dar.basis_
            basis.shape = (basis.shape[0], -1)
            product = np.dot(basis, basis.T)

            n_basis = product.shape[0]
            n_samples = _sigin.size

            # Test that the diagonal is constant if normalized
            ref = np.ones(n_basis) * n_samples
            if normalize:
                assert_array_almost_equal(product.flat[::n_basis + 1], ref)
            else:
                assert_array_not_almost_equal(product.flat[::n_basis + 1], ref)

            # Test that the rest is zero if orthogonalized
            product.flat[::n_basis + 1] = 0
            ref = np.zeros((n_basis, n_basis))
            if ortho:
                assert_array_almost_equal(product, ref)
            else:
                assert_array_not_almost_equal(product, ref)


def test_plot_comodulogram():
    # Smoke test with the standard plotting function
    model = fast_fitted_model()
    model.plot_lines()
    model.plot()
    plt.close('all')

    model = fast_fitted_model(sigdriv_imag=None)
    model.plot_lines()
    model.plot()
    plt.close('all')


def test_weighting_with_ones():
    # Test that weighting with ones is identical as no weighting
    factor = 1.5
    train_weights = np.ones_like(_sigin) * factor
    for klass in ALL_MODELS:
        model_2 = fast_fitted_model(klass=klass, train_weights=train_weights,
                                    test_weights=train_weights)
        model_1 = fast_fitted_model(klass=klass, train_weights=train_weights,
                                    test_weights=None)
        model_0 = fast_fitted_model(klass=klass, train_weights=None,
                                    test_weights=None)
        assert_array_almost_equal(model_1.AR_, model_0.AR_, decimal=5)
        assert_array_almost_equal(model_2.AR_, model_0.AR_, decimal=5)
        assert_array_almost_equal(model_1.G_, model_0.G_, decimal=5)
        assert_array_almost_equal(model_2.G_, model_0.G_, decimal=5)
        for train in [False, True]:
            assert_array_almost_equal(
                model_0._estimate_log_likelihood(train=train)[0] * factor,
                model_1._estimate_log_likelihood(train=train)[0], decimal=4)
            assert_array_almost_equal(
                model_1._estimate_log_likelihood(train=train)[0],
                model_2._estimate_log_likelihood(train=train)[0], decimal=4)


def test_weighting_with_zeros():
    # Test that
    split = int(n_points // 2)
    train_weights = np.ones_like(_sigin)
    train_weights[split:] = 0
    sigin_half = _sigin[:split]
    sigdriv_half = _sigdriv[:split]
    sigdriv_imag_half = _sigdriv_imag[:split]
    for klass in ALL_MODELS:
        print(klass.__name__)
        model_1 = fast_fitted_model(klass=klass, train_weights=train_weights)
        model_0 = fast_fitted_model(
            klass=klass, sigin=sigin_half, sigdriv=sigdriv_half,
            sigdriv_imag=sigdriv_imag_half, train_weights=None)
        assert_array_almost_equal(model_1.AR_, model_0.AR_)
        assert_array_almost_equal(model_1.G_, model_0.G_)
        for train in [False, True]:
            assert_array_almost_equal(
                model_0._estimate_log_likelihood(train=train),
                model_1._estimate_log_likelihood(train=train), decimal=5)
