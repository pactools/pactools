import numpy as np
import matplotlib.pyplot as plt

from pactools.modulation_index import modulation_index
from pactools.io.load_data_example import load_data_example


def example_comodulogram():
    """
    Example of a comodulogram, i.e. a plot of the MI for a grid of
    frequencies. The plot is automatically saved (cf. save_name), and the
    MI array is saved in csv.
    """
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    low_fq_width = 0.5  # Hz
    high_fq_width = 10.0  # Hz
    low_fq_range = np.arange(0.2, 5.2, 0.2)  # Hz
    high_fq_range = np.arange(0.2, 150., 5.0)  # Hz
    method = 'tort'  # 'ozkurt', 'tort', 'canolty'
    save_name = 'example_figure_saved'

    comodulogram = modulation_index(
        low_sig=sig, fs=fs, draw=True, method=method,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        save_name=save_name)

    print(comodulogram.shape)
    # save in csv
    np.savetxt('example_data_saved.csv', comodulogram, delimiter=', ')
    plt.show()


def example_phase_plot():
    """
    Example with only one MI computed for 1 couple of frequency.
    This example also show a plot of the mean amplitude of the fast oscillation
    for each phase of the slow oscillation.
    """
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    low_fq_width = 0.5  # Hz
    high_fq_width = 10.0  # Hz
    low_fq_range = [3.0]  # Hz
    high_fq_range = [85.0]  # Hz
    method = 'tort'  # 'ozkurt', 'tort', 'canolty'

    MI = modulation_index(
        low_sig=sig, fs=fs, draw=False, method=method,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        draw_phase=True)

    print('index = %.5f ' % MI)
    plt.show()

if __name__ == '__main__':

    example_comodulogram()
    # example_phase_plot()
