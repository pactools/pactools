import numbers

import numpy as np


def check_array(sig, dtype='float64', accept_none=False):
    if accept_none and sig is None:
        return None
    elif not accept_none and sig is None:
        raise ValueError("This array should not be None.")

    array = np.atleast_2d(np.array(sig, dtype=dtype))
    if array.ndim > 2:
        raise ValueError("This array should not be more than 2 dimensional.")
    return array


def check_consistent_shape(*args):
    for array in args:
        if array is not None and array.shape != args[0].shape:
            raise ValueError('Incompatible shapes. Got '
                             '(%s != %s)' % (array.shape, args[0].shape))


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
