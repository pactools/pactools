from pathlib import Path
from setuptools import setup

descr = """Estimation of phase-amplitude coupling (PAC) in neural time series,
           including with driven auto-regressive (DAR) models."""


# Get the version from the pactools module
version = None
lines = (Path(__file__).parent / "pactools" / "__init__.py").read_text("utf-8").splitlines()
for line in lines:
    if line.startswith("__version__"):
        version = line.split("=")[1].strip().strip('"')
        break
else:
    raise ValueError("Could not find version in pactools/__init__.py")

setup(
    name='pactools',
    version=version,
    description=descr,
    long_description=open('README.rst').read(),
    license='BSD (3-clause)',
    download_url='https://github.com/pactools/pactools.git',
    url='http://github.com/pactools/pactools',
    maintainer='Tom Dupre la Tour',
    maintainer_email='tom.dupre-la-tour@m4x.org',
    packages=[
        'pactools',
        'pactools.dar_model',
        'pactools.utils',
    ],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "mne",
        "h5py",
    ],
)
