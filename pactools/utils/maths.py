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
