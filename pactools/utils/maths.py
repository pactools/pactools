from itertools import chain

import numpy as np
from scipy import linalg


def norm(x):
    """Compute the Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). More precise than sqrt(squared_norm(x)).
    """
    x = np.asarray(x)

    if np.any(np.iscomplex(x)):
        return np.sqrt(squared_norm(x))
    else:
        nrm2, = linalg.get_blas_funcs(['nrm2'], [x])
        return nrm2(x)


def squared_norm(x):
    """Squared Euclidean or Frobenius norm of x.

    Returns the Euclidean norm when x is a vector, the Frobenius norm when x
    is a matrix (2-d array). Faster than norm(x) ** 2.
    """
    x = np.asarray(x)
    x = x.ravel(order='K')
    return np.dot(x, np.conj(x))


def square(c):
    """square a complex array"""
    return np.real(c * np.conjugate(c))


def argmax_2d(a):
    """Return the tuple (i, j) where a[i, j] = a.max()"""
    return np.unravel_index(np.argmax(a), a.shape)


def is_power2(num):
    """Test if number is a power of 2

    Parameters
    ----------
    num : int
        Number.

    Returns
    -------
    b : bool
        True if is power of 2.
    """
    num = int(num)
    return num != 0 and ((num & (num - 1)) == 0)


def compute_n_fft(signals):
    """
    Compute the number of element in the FFT to have a good prime
    decomposition, for hilbert filter.
    Parameters
    ----------
    signals : array

    Returns
    -------
    n_fft: integer >= signals.shape[-1]
    """
    n_fft = signals.shape[-1]
    while prime_factors(n_fft)[-1] > 20:
        n_fft += 1
    return n_fft


def prime_factors(num):
    """
    Decomposition in prime factor.
    Used to find signal length that speed up Hilbert transform.

    Parameters
    ----------
    num : int

    Returns
    -------
    decomposition : list of int
        List of prime factors sorted by ascending order
    """
    assert num > 0
    assert isinstance(num, int)

    decomposition = []
    for i in chain([2], range(3, int(np.sqrt(num)) + 1, 2)):
        while num % i == 0:
            num = num // i
            decomposition.append(i)
        if num == 1:
            break
    if num != 1:
        decomposition.append(num)
    return decomposition
