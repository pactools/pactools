# pactools

[![Build Status](https://travis-ci.org/pactools/pactools.svg?branch=master)](https://travis-ci.org/pactools/pactools) [![codecov](https://codecov.io/gh/pactools/pactools/branch/master/graph/badge.svg)](https://codecov.io/gh/pactools/pactools)

This package provides tools to estimate **phase-amplitude coupling (PAC)** in neural time series. In particular, it implements the **driven auto-regressive (DAR)** models presented in the reference below.

## Install

**Dependencies**

The code is not fully tested. It is supposed to be working with:

```
Python >= 3.5
NumPy >= 1.10
SciPy >= 0.17
Matplotlib >= 1.5
MNE-Python >= 0.11 (optional)
PYTEST >= 3.0 (optional, used for testing)
```

It might work with previous versions also, yet nothing is guaranteed.

**Install**

```
git clone https://github.com/pactools/pactools.git
cd pactools
make install
```

**Update**

```
git pull origin master
make in
```

## Cite

If you use this code in your project, please cite:

```
@inproceedings{duprelatour2017parametric,
  title={Parametric estimation of spectrum driven by an exogenous signal},
  author={Dupr{\'e} la Tour, Tom and Grenier, Yves and Gramfort, Alexandre},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
  pages={4301--4305},
  year={2017},
  organization={IEEE}
}
```
