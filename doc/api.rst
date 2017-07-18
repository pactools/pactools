.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: pactools

Phase amplitude coupling (PAC)
==============================

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: class.rst

   Comodulogram
   DelayEstimator
   PeakLocking
   :template: function.rst
   raw_to_mask
   simulate_pac


Driven auto-regressive (DAR) models
===================================

.. currentmodule:: pactools.dar_model

.. autosummary::
   :toctree: generated/
   :nosignatures:

   :template: class.rst
   DAR
   StableDAR
   :template: function.rst
   extract_driver


Utilities
=========

.. currentmodule:: pactools.utils

.. autosummary::
   :toctree: generated/
   :nosignatures:

   :template: class.rst
   fir.BandPassFilter
   fir.LowPassFilter
   spectrum.Spectrum
   spectrum.Coherence
   spectrum.Bicoherence
   :template: function.rst
   peak_finder.peak_finder
