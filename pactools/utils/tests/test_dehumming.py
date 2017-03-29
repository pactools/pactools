import numpy as np

from pactools.utils.dehummer import dehummer
from pactools.utils.maths import norm
from pactools.utils.testing import assert_greater
from pactools.utils.testing import assert_array_equal
from pactools.utils.testing import assert_raises


def test_dehumming_improve_snr():
    # Test that dehumming removes the added noise
    rng = np.random.RandomState(0)
    fs = 250.
    enf = 50.
    clean = rng.randn(512)
    noisy = clean.copy()

    # add ENF noise
    time = np.arange(512) / float(fs)
    for harmonic in (1, 2):
        noisy += np.sin(time * 2 * np.pi * enf * harmonic) / harmonic

    for dim in [(1, 512), (512, ), (512, 1)]:
        dehummed = dehummer(
            noisy.reshape(*dim), fs, enf=enf)
        assert_greater(norm(noisy - clean), norm(dehummed.ravel() - clean))
        assert_array_equal(dehummed.shape, dim)

    # test that the dehumming does not accept 2d arrays
    assert_raises(ValueError, dehummer, noisy.reshape(2, 256), fs)
