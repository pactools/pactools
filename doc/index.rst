.. pactools documentation master file, created by
   sphinx-quickstart on Thu Mar 23 10:56:51 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pactools's documentation!
====================================

This package provides tools to estimate **phase-amplitude coupling (PAC)** in neural time series.

In particular, it implements the **driven auto-regressive (DAR)** models presented in the reference below.

Installation
------------

We recommend the `Anaconda Python distribution <https://www.continuum.io/downloads>`_. To install ``pactools``, you first need to install its dependencies::

	$ pip install numpy matplotlib scipy

Then install pactools::

	$ pip install git+https://github.com/pactools/pactools.git#egg=pactools

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.

To check if everything worked fine, you can do::

	$ python -c 'import pactools'

and it should not give any error messages.


Content
-------

.. toctree::
   :maxdepth: 2

   api.rst


Cite
----

If you use this code in your project, please cite:

.. code::

    @inproceedings{duprelatour2017parametric,
     title={Parametric estimation of spectrum driven by an exogenous signal},
     author={Dupr{\'e} la Tour, Tom and Grenier, Yves and Gramfort, Alexandre},
     booktitle={Acoustics, Speech and Signal Processing (ICASSP), 2017 IEEE International Conference on},
     pages={4301--4305},
     year={2017},
     organization={IEEE}
     }



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
