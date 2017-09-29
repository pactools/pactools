import warnings


class _FakeParallel(object):
    """Useless decorator that is used like Parallel"""
    def __init__(self, n_jobs=1, *args, **kwargs):
        if n_jobs != 1:
            warnings.warn(
                "You have set n_jobs=%s to use parallel processing, "
                "but it has currently no effect. Please install joblib or "
                "scikit-learn to use it."
                % (n_jobs, ))

    def __call__(self, iterable):

        return list(iterable)


def _fake_delayed(func):
    return func


# try to import from joblib, otherwise, create a dummy function with no effect
try:
    from joblib import Parallel, delayed
except ImportError:
    try:
        from sklearn.externals.joblib import Parallel, delayed
    except ImportError:
        Parallel = _FakeParallel
        delayed = _fake_delayed
