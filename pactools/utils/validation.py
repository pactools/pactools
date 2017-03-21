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
        if array.shape != args[0].shape:
            raise ValueError('Incompatible shapes. Got '
                             '(%s != %s)' % (array.shape, args[0].shape))
