Welcome to pactools's documentation!
====================================

This package provides tools to estimate **phase-amplitude coupling (PAC)** in neural time series.

In particular, it implements the **driven auto-regressive (DAR)** models presented in the reference below.

WARNING: The API of this package is not fixed yet; function names and parameters may change without notice. Estimated date of reliable API: May 2017

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


.. toctree::
   :maxdepth: 2
   :hidden:

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

Acknowledgment
--------------

The project was supported by the ERC Starting Grant SLAB ERC-YStG-676943 to Alexandre Gramfort and IDEX Grant NoTime to Virginie van Wassenhove.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
