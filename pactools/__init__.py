from .bandpass_filter import multiple_band_pass
from .comodulogram import Comodulogram
from .references import REFERENCES
from .simulate_pac import simulate_pac, sigmoid
from .peak_locking import PeakLocking
from .mne_api import raw_to_mask


__version__ = '0.1'

__all__ = ['Comodulogram',
           'simulate_pac',
           'multiple_band_pass',
           'PeakLocking',
           'raw_to_mask',
           'REFERENCES',
           'sigmoid',
           ]
