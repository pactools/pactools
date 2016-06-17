import numpy as np
import matplotlib.pyplot as plt

from pactools.modulation_index import modulation_index
from pactools.plot_comodulogram import plot_comodulograms
from pactools.io.load_data_example import load_data_example
from pactools.dar_model import DAR, plot_dar_lines
from pactools.preprocess import extract


def example_dar():
    # load data
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    ordriv = 2  # typically 2
    ordar = 20  # typically in [10, 30]
    low_fq_width = 0.5  # Hz
    low_fq_range = [3.0]

    # extract the driver
    for low_sigs, high_sigs in extract(
            [sig], fs, low_fq_range=low_fq_range, bandwidth=low_fq_width):
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
    low_fq_range = np.arange(0.2, 5.2, 0.2)  # Hz
    high_fq_range = np.arange(0.2, 150., 4.0)  # Hz

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(16, 12))
    for i, method in enumerate(['ozkurt', 'tort', 'canolty',
                                DAR(ordriv=2, ordar=10)]):

        comodulogram = modulation_index(
            low_sig=sig, fs=fs, draw=False, method=method,
            low_fq_range=low_fq_range,
            low_fq_width=low_fq_width,
            high_fq_range=high_fq_range,
            high_fq_width=high_fq_width,
            n_surrogates=25)

        plot_comodulograms(comodulogram, fs, low_fq_range,
                           ['%s' % method], fig, [axs.ravel()[i]])

    fig.savefig('compare_comodulogram.png')
    plt.show()


if __name__ == '__main__':
    #example_dar()
    example_compare_comodulogram()
