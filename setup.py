import setuptools  # for python setup.py develop
from distutils.core import setup

setup(name='pactools',
      version='0.1',
      url='http://github.com/pactools/pactools',
      packages=['pactools',
                'pactools.dar_model',
                'pactools.utils',
                'pactools.viz',
                ],
      )
