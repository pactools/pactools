from __future__ import print_function
import warnings

import numpy as np
import scipy.io


def mat2sig(filename):
    """
    reads signal from matlab file

    returns data = {'sigs':sigs,'fs':fs}

    filename : name of file containing signal and sampling frequency

    if additionnal arguments have been saved by sig2mat, mat2sig restores
    them in data, e.g. {'sigs':sigs,'fs':fs,'fill':fill}
    """
    data = scipy.io.loadmat(filename, squeeze_me=True)
    return data


def sig2mat(filename, sigs, fs, **kwargs):
    """
    write signal to matlab file

    filename : name of the output file
    sigs     : list of signal that will be saved
    fs       : sampling frequency

    additionnal arguments are stored in file as (key, value) pairs
    """
    if not isinstance(sigs, list):
        sigs = [sigs]
    sigs = [sig.astype(np.float32) for sig in sigs]

    with warnings.catch_warnings():  # avoid warning in savemat
        warnings.simplefilter("ignore")
        data = {'sigs': sigs, 'fs': fs}
        for key, value in kwargs.items():
            data[key] = value
        scipy.io.savemat(filename, data)
