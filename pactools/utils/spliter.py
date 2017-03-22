import numpy as np


class Spliter(object):
    """
    Split a signal into several blocks
    """

    def __init__(self, n_samples, fs, t_split, t_step=None):
        self.n_samples = n_samples
        self.fs = float(fs)
        self.t_split = t_split
        if t_step is None:
            self.t_step = t_split
        else:
            self.t_step = t_step

        self.n_splits = 0

    def __iter__(self):
        n_samples = self.n_samples
        fs = self.fs

        start = 0
        stop = int(self.t_split * fs)
        step = int(self.t_step * fs)

        while stop <= n_samples:
            yield start, stop
            self.n_splits += 1

            start += step
            stop += step


def blend_and_ravel(sig, n_blend):
    """Ravel a 2-dimensional array, with some blending at the edges"""
    n_blend = int(n_blend)
    if n_blend > 0:
        triangle = np.linspace(0, 1, n_blend)
        sig[1:, :n_blend] *= triangle
        sig[:-1, -n_blend:] *= triangle[::-1]
        sig[:-1, -n_blend:] += sig[1:, :n_blend]
        sig = sig[:, n_blend:]

    sig = sig.ravel()
    return sig
