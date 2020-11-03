=============================
Getting Started with pactools
=============================

.. image:: https://travis-ci.org/pactools/pactools.svg?branch=master
    :target: https://travis-ci.org/pactools/pactools
    :alt: Build Status

.. image:: https://codecov.io/gh/pactools/pactools/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pactools/pactools
    :alt: Test coverage

.. image:: https://img.shields.io/badge/python-2.7-blue.svg
    :target: https://github.com/pactools/pactools
    :alt: Python27

.. image:: https://img.shields.io/badge/python-3.6-blue.svg
    :target: https://github.com/pactools/pactools
    :alt: Python36

This package provides tools to estimate **phase-amplitude coupling (PAC)**
in neural time series.

In particular, it implements the **driven auto-regressive (DAR)**
models presented in the reference below [`Dupre la Tour et al. 2017`_].

Read more in the `documentation <https://pactools.github.io>`_.

Installation
============

To install ``pactools``, use one of the following two commands:

- Latest stable version::

    pip install pactools

- Development version::

    pip install git+https://github.com/pactools/pactools.git#egg=pactools

To upgrade, use the ``--upgrade`` flag provided by ``pip``.

To check if everything worked fine, you can do::

	python -c 'import pactools'

and it should not give any error messages.

Phase-amplitude coupling (PAC)
==============================
Among the different classes of cross-frequency couplings,
phase-amplitude coupling (PAC) - i.e. high frequency activity time-locked
to a specific phase of slow frequency oscillations - is by far the most
acknowledged.
PAC is typically represented with a comodulogram, which shows the strenght of
the coupling over a grid of frequencies.
Comodulograms can be computed in `pactools` with more
than 10 different methods.

.. include:: generated/backreferences/pactools.Comodulogram.examples
.. raw:: html

    <div style='clear:both'></div>

Driven auto-regressive (DAR) models
===================================
One of the method is based on driven auto-regressive (DAR) models.
As this method models the entire spectrum simultaneously, it avoids the
pitfalls related to incorrect filtering or the use of the Hilbert transform
on wide-band signals. As the model is probabilistic, it also provides a
score of the model **goodness of fit** via the likelihood, enabling easy
and legitimate model selection and parameter comparison;
this data-driven feature is unique to such model-based approach.

We recommend using DAR models to estimate PAC in neural time-series.
More detail in [`Dupre la Tour et al. 2017`_].


.. include:: generated/backreferences/pactools.dar_model.DAR.examples
.. raw:: html

    <div style='clear:both'></div>

Acknowledgment
==============

This work was supported by the ERC Starting Grant SLAB ERC-YStG-676943 to
Alexandre Gramfort, the ERC Starting Grant MindTime ERC-YStG-263584 to Virginie
van Wassenhove, the ANR-16-CE37-0004-04 AutoTime to Valerie Doyere and Virginie
van Wassenhove, and the Paris-Saclay IDEX NoTime to Valerie Doyere, Alexandre
Gramfort and Virginie van Wassenhove,

Cite this work
==============

If you use this code in your project, please cite
[`Dupre la Tour et al. 2017`_]:


.. code-block::

    @article{duprelatour2017nonlinear,
        author = {Dupr{\'e} la Tour, Tom and Tallot, Lucille and Grabot, Laetitia and Doy{\`e}re, Val{\'e}rie and van Wassenhove, Virginie and Grenier, Yves and Gramfort, Alexandre},
        journal = {PLOS Computational Biology},
        publisher = {Public Library of Science},
        title = {Non-linear auto-regressive models for cross-frequency coupling in neural time series},
        year = {2017},
        month = {12},
        volume = {13},
        url = {https://doi.org/10.1371/journal.pcbi.1005893},
        pages = {1-32},
        number = {12},
        doi = {10.1371/journal.pcbi.1005893}
    }


.. _Dupre la Tour et al. 2017: http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005893
