from .bandpass_filter import multiple_band_pass
from .comodulogram import Comodulogram, REFERENCES
from .simulate_pac import simulate_pac, sigmoid
from .peak_locking import PeakLocking


__version__ = '0.1'

__all__ = ['Comodulogram',
           'simulate_pac',
           'multiple_band_pass',
           'PeakLocking',
           'REFERENCES',
           'sigmoid',
           ]
