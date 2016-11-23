import numpy as np
import matplotlib.pyplot as plt

from pactools.comodulogram import comodulogram
from pactools.io.load_data_example import load_data_example
from pactools.dar_model import DAR


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

    comod = comodulogram(
        low_sig=sig, fs=fs, draw=True, method=method,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        save_name=save_name)

    print(comod.shape)
    # save in csv
    np.savetxt('example_data_saved.csv', comod, delimiter=', ')
    plt.show()


def example_comodulogram_with_mask():
    """
    Example of a comodulogram computed with a mask.
    """
    # here the epochs are not concatenated
    sig, fs = load_data_example(ravel=False)
    print('n_epochs, n_points = %s' % (sig.shape,))

    # Parameters for the PAC analysis
    low_fq_width = 0.5  # Hz
    high_fq_width = 10.0  # Hz
    low_fq_range = np.arange(0.2, 5.2, 0.2)  # Hz
    high_fq_range = np.arange(0.2, 150., 5.0)  # Hz
    method = 'tort'  # 'ozkurt', 'tort', 'canolty'
    method = DAR(ordar=10, ordriv=1)
    save_name = 'example_figure_saved'

    # here we use masks to select the time windows
    mask_list = []
    tmin = 0.0
    for t_start, t_stop in [[1, 4], [4, 7], [7, 10],
                            [10, 13], [13, 16], [16, 19],
                            [19, 22], [22, 25], [25, 28]]:
        mask = np.ones(sig.shape)
        start = int((t_start - tmin) * fs)
        stop = int((t_stop - tmin) * fs)
        mask[:, start:stop] = 0

        mask_list.append(mask)

    comod_list = comodulogram(
        low_sig=sig, fs=fs, draw=True, method=method,
        mask=mask_list,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        save_name=save_name)

    print("lenght of the comodulogram list = %s" % len(comod_list))
    plt.show()


def example_phase_plot():
    """
    Example with only one modulation_index computed for 1 couple of frequency.
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

    modulation_index = comodulogram(
        low_sig=sig, fs=fs, draw=False, method=method,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_range=high_fq_range,
        high_fq_width=high_fq_width,
        draw_phase=True)

    print('modulation index = %.5f ' % modulation_index)
    plt.show()

if __name__ == '__main__':

    # example_comodulogram()
    example_comodulogram_with_mask()
    # example_phase_plot()
    pass
