from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import mne

from pactools.modulation_index import modulation_index
from pactools.utils.spliter import Spliter, blend_and_ravel
from pactools.plot_comodulogram import plot_comodulograms


raw_file_name = 'LTM1_L1-L2'
dirname = '/data/tdupre/rat_data/' + raw_file_name + '/'


def example_channel_by_channel():
    channels = ['c%d' % c for c in [0, 1, 2, 3, 4, 5, 6, 7]]
    for channel in channels:
        print('channel %s' % channel)
        # load the 6 files, and crop the ERP
        data_list = []
        filename_full_list = []
        for event in ['_e-1_', '_e1_']:
            segment_in = [-30.0, 0.0, 30.0, 60.0]
            parts = []
            for i in range(len(segment_in) - 1):
                part = '[%.1f-%.1f]' % (segment_in[i], segment_in[i + 1])
                filename_full = channel + event + part + '-epo.fif'
                loaded_epochs = mne.read_epochs(dirname + filename_full)
                fs = loaded_epochs.info['sfreq']
                data = loaded_epochs.get_data()

                #remove the first t_evok seconds with the ERP
                t_evok = 1.0
                data = data[:, :, int(t_evok * fs):]

                split = Spliter(data.shape[-1], fs, t_split=9.0)  # ex: 9, 29
                for start, stop in split:
                    data_list.append(data[:, :, start:stop])
                    split_part = ('[%.1f,%.1f]' %
                                  (segment_in[i] + t_evok + start / fs,
                                   segment_in[i] + t_evok + stop / fs))
                    filename_full_list.append(raw_file_name + '_' +
                                              channel + event + split_part)
                    parts.append(split_part)

        # take the same number of epochs for each comodulogram
        n_epochs_min = min([d.shape[0] for d in data_list])

        # not enough clean epochs
        if n_epochs_min <= 1:
            continue

        # general parameters
        low_fq_range = np.arange(1.0, 5.1, 0.1)
        low_fq_width = 0.5
        high_fq_range = np.arange(0.2, 150., 5.0)
        high_fq_width = 12.0

        # model and comodulogram parameters
        method = 'tort'
        save_name_suffix = '_%s' % method

        # computes the comodulograms
        comodulograms = []
        for data, filename_full in zip(data_list, filename_full_list):
            # blend the epochs together
            t_blend = 0.1
            data = blend_and_ravel(data[:n_epochs_min, 0], int(t_blend * fs))

            comodulogram = modulation_index(
                data, fs=fs, draw=False, method=method,
                low_fq_range=low_fq_range,
                low_fq_width=low_fq_width,
                high_fq_range=high_fq_range,
                high_fq_width=high_fq_width,
                save_name=filename_full + save_name_suffix,
                progress_bar=True)

            comodulograms.append(comodulogram)

        # plot the comodulograms
        fig, axs = plt.subplots(2, len(comodulograms) // 2,
                                sharex=True, sharey=True,
                                figsize=(1.4 * len(comodulograms), 5.))
        titles = ['cs- %s' % (parts[i]) for i in range(len(parts))]
        titles += ['cs+ %s' % (parts[i]) for i in range(len(parts))]
        plot_comodulograms(comodulograms, fs, low_fq_range, titles, fig, axs)

        # add suptitle
        fig.subplots_adjust(top=0.85)
        name = raw_file_name + '_' + channel + save_name_suffix
        fig.suptitle('Comodulograms, %s, driver low_fq_width'
                     ' = %.2f Hz' % (name, low_fq_width), fontsize=14)

        # better ticks in x axis
        xlim = axs.ravel()[0].get_xlim()
        ticks = np.linspace(xlim[0], xlim[1], int(xlim[1] - xlim[0]) + 1)
        axs.ravel()[0].set_xticks(ticks)

        fig.savefig(name + '.png')
        plt.close('all')


def example_all_channels():
    # select some channels
    channels = ['0', '1', '2', '3']

    # list of the splits over time
    #splits = [[-29, 0], [1, 30], [30, 60]]
    splits = [[-29, -20], [-19, -10], [-9, 0],
              [1, 10], [11, 20], [21, 30],
              [30, 39], [40, 49], [50, 59]]

    data_list = []
    filename_full_list = []
    for event in ['e-1_', 'e1_']:
        tmin = -30.0
        tmax = 60.0
        segment = '[%.1f-%.1f]' % (tmin, tmax)
        filename_full = event + segment + '-epo.fif'

        loaded_epochs = mne.read_epochs(dirname + filename_full)
        loaded_epochs.pick_channels(channels)
        fs = loaded_epochs.info['sfreq']
        data = loaded_epochs.get_data()

        n_epochs, n_channels, n_times = data.shape

        parts = []
        for t_start, t_stop in splits:
            start = int((t_start - tmin) * fs)
            stop = int((t_stop - tmin) * fs)
            data_list.append(data[:, :, start:stop])
            split_part = '[%.1f,%.1f]' % (t_start, t_stop)
            filename_full_list.append(raw_file_name + '_' +
                                      event + split_part)
            parts.append(split_part)

    # take the same number of epochs for each comodulogram
    n_epochs_min = min([d.shape[0] for d in data_list])

    # not enough clean epochs
    if n_epochs_min <= 1:
        return None

    # general parameters
    low_fq_range = np.arange(1.0, 5.1, 0.1)
    low_fq_width = 0.5
    high_fq_range = np.arange(0.2, 150., 5.0)
    high_fq_width = 12.0

    # model and comodulogram parameters
    method = 'tort'
    save_name_suffix = '_%s' % method

    for i_channel, channel in enumerate(channels):
        print('channel %s' % channel)

        # computes the comodulograms
        comodulograms = []
        for data, filename_full in zip(data_list, filename_full_list):
            # blend the epochs together
            t_blend = 0.1
            data = blend_and_ravel(data[:n_epochs_min, i_channel],
                                   int(t_blend * fs))

            comodulogram = modulation_index(
                data, fs=fs, draw=False, method=method,
                low_fq_range=low_fq_range,
                low_fq_width=low_fq_width,
                high_fq_range=high_fq_range,
                high_fq_width=high_fq_width,
                save_name=filename_full + save_name_suffix,
                progress_bar=True)

            comodulograms.append(comodulogram)

        # plot the comodulograms
        fig, axs = plt.subplots(2, len(comodulograms) // 2,
                                sharex=True, sharey=True,
                                figsize=(1.4 * len(comodulograms), 5.))
        titles = ['cs- %s' % (parts[i]) for i in range(len(parts))]
        titles += ['cs+ %s' % (parts[i]) for i in range(len(parts))]
        plot_comodulograms(comodulograms, fs, low_fq_range, titles, fig, axs)

        # add suptitle
        fig.subplots_adjust(top=0.85)
        name = raw_file_name + '_' + channel + save_name_suffix
        fig.suptitle('Comodulograms, %s, driver low_fq_width'
                     ' = %.2f Hz' % (name, low_fq_width), fontsize=14)

        # better ticks in x axis
        xlim = axs.ravel()[0].get_xlim()
        ticks = np.linspace(xlim[0], xlim[1], int(xlim[1] - xlim[0]) + 1)
        axs.ravel()[0].set_xticks(ticks)

        fig.savefig(name + '.png')
        print('Saved: %s' % (name + '.png'))
        plt.close('all')

if __name__ == '__main__':
    example_all_channels()
