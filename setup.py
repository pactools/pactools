import setuptools  # used with `python setup.py develop`
from distutils.core import setup

descr = """Estimation of phase-amplitude coupling (PAC) in neural time series,
           including with driven auto-regressive (DAR) models."""

setup(
    name='pactools',
    version='0.1',
    description=descr,
    long_description=open('README.md').read(),
    license='BSD (3-clause)',
    download_url='https://github.com/pactools/pactools.git',
    url='http://github.com/pactools/pactools',
    maintainer='Tom Dupre la Tour',
    maintainer_email='tom.dupre-la-tour@m4x.org',
    packages=[
        'pactools',
        'pactools.dar_model',
        'pactools.utils',
    ], )
