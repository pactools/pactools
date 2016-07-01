import numpy as np
import matplotlib.pyplot as plt

from pactools.phase_locking import time_frequency_peak_locking
from pactools.io.load_data_example import load_data_example


def example_phase_locking():
    """
    Example of a theta-trough locked Time-frequency plot of mean power
    modulation time-locked to the theta trough
    """
    sig, fs = load_data_example()

    # Parameters for the PAC analysis
    low_fq_width = 2.0  # Hz
    high_fq_width = 8.0  # Hz
    low_fq = 3.4  # Hz
    high_fq_range = np.linspace(2.0, 160.0, 40)  # Hz
    save_name = 'example_figure_saved'

    time_frequency_peak_locking(sig=sig, fs=fs,
                                low_fq=low_fq,
                                low_fq_width=low_fq_width,
                                high_fq_range=high_fq_range,
                                high_fq_width=high_fq_width,
                                t_plot=1.0,
                                save_name=save_name)

    plt.show()


if __name__ == '__main__':

    example_phase_locking()
