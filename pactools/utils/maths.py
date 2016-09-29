import numbers

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
    x = x.ravel(order='K')
    return np.dot(x, np.conj(x))


def argmax_2d(a):
    return np.unravel_index(np.argmax(a), a.shape)


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
