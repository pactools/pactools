from .comodulogram import Comodulogram, REFERENCES

from .create_signal import create_signal, sigmoid

from .peak_locking import PeakLocking

from .preprocess import decimate, extract_and_fill
from .preprocess import low_pass_and_fill, whiten, fill_gap
from .preprocess import extract_driver

__version__ = '0.1'

__all__ = ['Comodulogram',
           'create_signal',
           'decimate',
           'extract_driver',
           'extract_and_fill',
           'fill_gap',
           'low_pass_and_fill',
           'PeakLocking',
           'REFERENCES',
           'sigmoid',
           'whiten',
           ]
