from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import mne

from pactools.utils.spliter import blend_and_ravel
from pactools.modulation_index import modulation_index
from pactools.plot_comodulogram import plot_comodulograms


def example_on_each_epoch():
    # load data
    dirname = '/data/tdupre/'
    filename = 'S08_objPerf_R1_onset_ShCoLg_Block123-epo.fif'
    loaded_epochs = mne.read_epochs(dirname + filename)

    # choose the channels
    channel_names = ['MEG2022', ]
    chosen_epochs = loaded_epochs.pick_channels(channel_names)

    # extract the data (for now, the functions do not accept MNE objects)
    fs = chosen_epochs.info['sfreq']
    sig = chosen_epochs.get_data()
    n_epochs, n_channels, n_points = sig.shape

    for i_channel in range(n_channels):
        channel_signal = sig[:, i_channel, :]

        for epoch in range(n_epochs):
            save_name = '%s_%d' % (channel_names[i_channel], epoch + 1)
            one_sig = channel_signal[epoch].ravel()

            # array of frequency for phase signal
            low_fq_range = np.arange(0.1, 5.1, 0.1)
            low_fq_width = 0.5  # Hz
            # array of frequency for amplitude signal
            high_fq_range = np.arange(1, fs / 2, 3)
            high_fq_width = 10.0  # Hz

            comods = []
            titles = []
            for method in ['tort', 'ozkurt']:
                # compute the comodulograms
                comod = modulation_index(
                    low_sig=one_sig, fs=fs, draw=False, method=method,
                    low_fq_range=low_fq_range, low_fq_width=low_fq_width,
                    high_fq_range=high_fq_range, high_fq_width=high_fq_width,
                    vmin=None, vmax=None)
                comods.append(comod)
                titles.append(method)

            # plot the results
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))

            for i, comod in enumerate(comods):
                plot_comodulograms(comod, fs, low_fq_range, [titles[i]],
                                   fig, [axs[i]], plt.get_cmap('viridis'),
                                   None, None)

            # save the figure
            fig.savefig(save_name)
            plt.close('all')


def example_concatenating_epochs():
    # load data
    dirname = '/data/tdupre/'
    filename = 'S08_objPerf_R1_onset_ShCoLg_Block123-epo.fif'
    loaded_epochs = mne.read_epochs(dirname + filename)

    # choose the channels
    channel_names = ['MEG2022', 'MEG0232', 'MEG1821',
                     'EEG002', 'EEG007', 'EEG015']
    # channel_names = ['MEG2021', 'MEG0231']
    # channel_names = loaded_epochs.info['ch_names']
    chosen_epochs = loaded_epochs.pick_channels(channel_names)

    # extract the data (for now, the functions do not accept MNE objects)
    fs = chosen_epochs.info['sfreq']
    sig = chosen_epochs.get_data()
    n_epochs, n_channels, n_points = sig.shape

    for i_channel in range(n_channels):
        channel_signal = sig[:, i_channel, :]

        # for epoch in range(n_epochs):
        #     save_name = '%s_%d' % (channel_names[i_channel], epoch + 1)
        #     one_sig = channel_signal[epoch].ravel()

        # concatenante all epochs and blend to smooth transitions
        t_blend = 0.3  # sec
        one_sig = blend_and_ravel(channel_signal, int(fs * t_blend))
        save_name = '%s' % (channel_names[i_channel])

        # array of frequency for phase signal
        low_fq_range = np.arange(0.1, 5.1, 0.1)
        low_fq_width = 0.5  # Hz
        # array of frequency for amplitude signal
        high_fq_range = np.arange(1, fs / 2, 3)
        high_fq_width = 10.0  # Hz

        comods = []
        titles = []
        for method in ['tort', 'ozkurt']:
            # compute the comodulograms
            comod = modulation_index(
                low_sig=one_sig, fs=fs, draw=False, method=method,
                low_fq_range=low_fq_range, low_fq_width=low_fq_width,
                high_fq_range=high_fq_range, high_fq_width=high_fq_width,
                vmin=None, vmax=None)
            comods.append(comod)
            titles.append(method)

        # plot the results
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))

        # loop is not necessary if we have comodulogram with the same scales
        for i, comod in enumerate(comods):
            plot_comodulograms(comod, fs, low_fq_range, [titles[i]],
                               fig, [axs[i]], plt.get_cmap('viridis'),
                               None, None)

        # save the figure
        fig.savefig(save_name)
        plt.close('all')


if __name__ == '__main__':
    example_concatenating_epochs()
    example_on_each_epoch()
