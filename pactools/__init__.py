from .bandpass_filter import multiple_band_pass
from .comodulogram import Comodulogram, REFERENCES
from .create_signal import create_signal, sigmoid
from .peak_locking import PeakLocking


__version__ = '0.1'

__all__ = ['Comodulogram',
           'create_signal',
           'multiple_band_pass',
           'PeakLocking',
           'REFERENCES',
           'sigmoid',
           ]
