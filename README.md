# pactools

[![Build Status](https://travis-ci.org/pactools/pactools.svg?branch=master)](https://travis-ci.org/pactools/pactools) [![codecov](https://codecov.io/gh/pactools/pactools/branch/master/graph/badge.svg)](https://codecov.io/gh/pactools/pactools) [![Python27](https://img.shields.io/badge/python-2.7-blue.svg)](https://github.com/pactools/pactools) [![Python36](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/pactools/pactools)

This package provides tools to estimate **phase-amplitude coupling (PAC)** in neural time series.

In particular, it implements the **driven auto-regressive (DAR)** models presented in the reference below.

Read more in the [documentation](https://pactools.github.io).

## Install

We recommend the [Anaconda Python distribution](https://www.continuum.io/downloads). To install `pactools`, you first need to install its dependencies::

```console
pip install numpy matplotlib scipy
```

Then install pactools::

```console
pip install git+https://github.com/pactools/pactools.git#egg=pactools
```

If you do not have admin privileges on the computer, use the `--user` flag with `pip`. To upgrade, use the `--upgrade` flag provided by `pip`.

To check if everything worked fine, you can do::

```console
python -c 'import pactools'
```

and it should not give any error messages.

## Acknowledgment

The project was supported by the ERC Starting Grant SLAB ERC-YStG-676943 to Alexandre Gramfort and IDEX Grant NoTime to Virginie van Wassenhove.

## Cite

If you use this code in your project, please cite [this paper](https://hal.archives-ouvertes.fr/hal-01448603v2):

```
@article {duprelatour159731,
    author = {Dupr\'e la Tour, Tom and Tallot, Lucille and Grabot, Laetitia and Doy\`ere, Val\'erie and van Wassenhove, Virginie and Grenier, Yves and Gramfort, Alexandre},
    title = {Non-linear Auto-Regressive Models for Cross-Frequency Coupling in Neural Time Series},
    year = {2017},
    doi = {10.1101/159731},
    publisher = {Cold Spring Harbor Labs Journals},
    URL = {http://www.biorxiv.org/content/early/2017/07/06/159731},
    journal = {bioRxiv}
}
```
