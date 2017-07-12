.. _api_documentation:

=================
API Documentation
=================

.. currentmodule:: pactools

Phase amplitude coupling (PAC)
==============================

.. autosummary::
   :toctree: generated/

   Comodulogram
   DelayEstimator
   PeakLocking
   raw_to_mask
   simulate_pac


Driven auto-regressive (DAR) models
===================================

.. currentmodule:: pactools.dar_model

.. autosummary::
   :toctree: generated/

   DAR
   StableDAR
   extract_driver


Utilities
=========

.. currentmodule:: pactools.utils

.. autosummary::
   :toctree: generated/

   fir.BandPassFilter
   fir.LowPassFilter
   spectrum.Spectrum
   spectrum.Coherence
   spectrum.Bicoherence
   peak_finder.peak_finder
