import numpy as np
import matplotlib.pyplot as plt

from pactools.comodulogram import comodulogram
from pactools.plot_comodulogram import plot_comodulogram
from pactools.io.load_data_example import load_data_example
from pactools.dar_model import DAR, plot_dar_lines
from pactools.preprocess import extract


def example_dar():
    # load data
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    ordriv = 2  # typically in [0, 2]
    ordar = 20  # typically in [4, 30]
    low_fq_width = 0.5  # Hz
    low_fq_range = [3.0]

    # extract the driver
    for low_sigs, high_sigs in extract(
            [sig], fs,
            low_fq_range=low_fq_range,
            bandwidth=low_fq_width,
            whitening=None):
        sigdriv = low_sigs[0]
        sigin = high_sigs[0]

    # fit the DAR model
    model = DAR(ordriv=ordriv, ordar=ordar)
    model.fit(sigin=sigin, sigdriv=sigdriv, fs=fs)

    plot_dar_lines(model)
    plt.show()


def example_compare_comodulogram():
    """
    Example of a comodulogram, i.e. a plot of the MI for a grid of
    frequencies, with 4 different methods.
    """
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    low_fq_width = 0.5  # Hz
    high_fq_width = 12.0  # Hz
    low_fq_range = np.arange(1.0, 6.2, 0.2)  # Hz
    high_fq_range = np.arange(10., fs / 2, 4.0)  # Hz

    # Define two models DAR:
    dar_model = DAR(ordriv=1, ordar=12, bic=False)
    # ordriv: order of the polynomial expansion of the driver.
    #         Typically in [0, 2]. With 0, no PAC is present in the model
    # ordar : order of the AutoRegressive part.
    #         Typically in [4, 30]. A low value will give a smooth Power
    #         Spectral Density.
    # bic   : if True, a grid search is performed on both parameters
    #         ordriv (in [0, ordriv]) and ordar (in [0, ordar]).
    #         The couple of parameters with the best BIC will be selected.
    #         The selection is done independently for all driver's frequencies.
    #         If the signal is short, low values of (ordar, ordriv) will
    #         be selected.

    n_surrogates = 0
    contours = 4.0 if n_surrogates > 1 else None

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 12))

    # The model is given to the 'method' parameter.
    for i, method in enumerate(['tort', 'ozkurt', 'canolty', dar_model]):

        comod = comodulogram(
            low_sig=sig, fs=fs, draw=False, method=method,
            low_fq_range=low_fq_range,
            low_fq_width=low_fq_width,
            high_fq_range=high_fq_range,
            high_fq_width=high_fq_width,
            n_surrogates=n_surrogates)

        plot_comodulogram(comod, fs, low_fq_range, high_fq_range,
                          ['%s' % method], fig, [axs.ravel()[i]],
                          contours=contours)

    fig.savefig('compare_comodulogram_%ds.png' % n_surrogates)
    plt.show()


if __name__ == '__main__':
    # example_dar()
    example_compare_comodulogram()
