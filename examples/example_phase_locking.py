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

    time_frequency_peak_locking(low_sig=sig, fs=fs,
                                low_fq=low_fq,
                                low_fq_width=low_fq_width,
                                high_fq_range=high_fq_range,
                                high_fq_width=high_fq_width,
                                t_plot=1.0,
                                save_name=save_name)

    plt.show()


def example_phase_locking_with_mask():
    """
    Example of a comodulogram computed with a mask.
    """
    # here the epochs are not concatenated
    sig, fs = load_data_example(ravel=False)
    print('n_epochs, n_points = %s' % (sig.shape,))

    # Parameters for the PAC analysis
    low_fq_width = 2.0  # Hz
    high_fq_width = 8.0  # Hz
    low_fq = 3.4  # Hz
    high_fq_range = np.linspace(2.0, 160.0, 40)  # Hz

    tmin = 0.0  # the loaded epoch start at tmin=0, i.e. the CS event
    for t_start, t_stop in [[1, 10], [10, 19], [19, 28]]:
        mask = np.zeros(sig.shape)
        start = int((t_start - tmin) * fs)
        stop = int((t_stop - tmin) * fs)
        mask[:, start:stop] = 1
        save_name = 'example_phase_locking_[%s,%s]' % (t_start, t_stop)

        time_frequency_peak_locking(low_sig=sig, fs=fs,
                                    mask=mask,
                                    low_fq=low_fq,
                                    low_fq_width=low_fq_width,
                                    high_fq_range=high_fq_range,
                                    high_fq_width=high_fq_width,
                                    t_plot=1.0, draw_peaks=False,
                                    save_name=save_name)
    plt.show()


if __name__ == '__main__':

    # example_phase_locking()
    example_phase_locking_with_mask()
