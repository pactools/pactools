from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import mne

from pactools.comodulogram import comodulogram, get_maximum_pac
from pactools.utils.spliter import Spliter, blend_and_ravel
from pactools.plot_comodulogram import plot_comodulograms
from pactools.utils.progress_bar import ProgressBar
from pactools.dar_model import DAR


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

            comod = comodulogram(
                low_sig=data, fs=fs, draw=False, method=method,
                low_fq_range=low_fq_range,
                low_fq_width=low_fq_width,
                high_fq_range=high_fq_range,
                high_fq_width=high_fq_width,
                save_name=filename_full + save_name_suffix,
                progress_bar=True)

            comodulograms.append(comod)

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
    # select some channels in ['0', '1', '2', '3', '4', '5', '6', '7']
    channels = ['0', '1', '2', '3']
    # select some events in ['e-1_', 'e1_']
    events = ['e-1_', 'e1_']

    # list of the splits over time
    #splits = [[-29, 0], [1, 30], [30, 60]]
    splits = [[-29, -20], [-19, -10], [-9, 0],
              [1, 10], [11, 20], [21, 30],
              [30, 39], [40, 49], [50, 59]]

    data_list, filename_full_list, parts, fs = load_data_and_splits(
        events, channels, splits)
    n_epochs, n_channels, n_times = data_list[0].shape

    # take the same number of epochs for each comodulogram
    n_epochs_min = min([d.shape[0] for d in data_list])
    # not enough clean epochs
    if n_epochs_min <= 1:
        return None

    # filtering parameters
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

            comod = comodulogram(
                low_sig=data, fs=fs, draw=False, method=method,
                low_fq_range=low_fq_range,
                low_fq_width=low_fq_width,
                high_fq_range=high_fq_range,
                high_fq_width=high_fq_width,
                save_name=filename_full + save_name_suffix,
                progress_bar=True)

            comodulograms.append(comod)

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


def load_data_and_splits(events, channels, splits):
    tmin, tmax = -30.0, 60.0
    segment = '[%.1f-%.1f]' % (tmin, tmax)

    data_list = []
    filename_full_list = []
    for event in events:
        filename_full = event + segment + '-epo.fif'

        loaded_epochs = mne.read_epochs(dirname + filename_full)
        loaded_epochs.pick_channels(channels)
        fs = loaded_epochs.info['sfreq']
        data = loaded_epochs.get_data()

        parts = []
        for t_start, t_stop in splits:
            start = int((t_start - tmin) * fs)
            stop = int((t_stop - tmin) * fs)
            data_list.append(data[:, :, start:stop])
            split_part = '[%.1f,%.1f]' % (t_start, t_stop)
            filename_full_list.append(raw_file_name + '_' +
                                      event + split_part)
            parts.append(split_part)

    return data_list, filename_full_list, parts, fs


def example_all_channels_maximum_pac():
    # select some channels in ['0', '1', '2', '3', '4', '5', '6', '7']
    channels_low = ['1', '1', '1']  # phase signal (low frequency)
    channels_high = ['1', '2', '3']  # amplitude signal (high frequency)
    # select some events in ['e-1_', 'e1_']
    events = ['e-1_', 'e1_']
    # select the number of epochs (9 for all, 0 for merging them)
    epochs = 0

    # filtering parameters
    low_fq_range = np.arange(1., 5.2, 0.2)
    low_fq_width = 0.5
    high_fq_range = np.arange(60., 110., 5.0)
    high_fq_width = 12.0
    method = 'tort'
    method = DAR(ordar=10, ordriv=1)

    # splits over time
    t_start, t_stop = -30., 60.  # sec
    t_split, t_step = 3., 1.0  # sec
    splits = np.c_[np.arange(t_start, t_stop - t_split, t_step),
                   np.arange(t_start, t_stop - t_split, t_step) + t_split]
    n_splits = splits.shape[0]

    all_channels = list(set(channels_low + channels_high))
    all_channels.sort()
    data_list, filename_full_list, _, fs = load_data_and_splits(
        events, all_channels, splits)
    n_epochs, n_channels, n_times = data_list[0].shape

    # take the same number of epochs for each comodulogram
    n_epochs_min = min([d.shape[0] for d in data_list])
    # not enough clean epochs
    if n_epochs_min <= 1:
        return None

    save_name_suffix = '_%s' % method

    for channel_low, channel_high in zip(channels_low, channels_high):
        print('channel %s and %s' % (channel_low, channel_high))
        i_channel_low = all_channels.index(channel_low)
        i_channel_high = all_channels.index(channel_high)

        # computes the comodulograms
        results = []
        bar = ProgressBar(title='max PAC over time', max_value=len(data_list))
        for data, filename_full in zip(data_list, filename_full_list):

            if epochs == 0:
                # blend the epochs together
                t_blend = 0.1
                data_low = blend_and_ravel(data[:n_epochs_min, i_channel_low],
                                           int(t_blend * fs))[None, :]
                data_high = blend_and_ravel(data[:n_epochs_min,
                                                 i_channel_high],
                                            int(t_blend * fs))[None, :]
                n_epochs = 1
            else:
                n_epochs_min = min(n_epochs_min, epochs)
                data_low = data[:n_epochs_min, i_channel_low]
                data_high = data[:n_epochs_min, i_channel_high]
                n_epochs = n_epochs_min

            for epoch_data_low, epoch_data_high in zip(data_low, data_high):
                comod = comodulogram(
                    low_sig=epoch_data_low, high_sig=epoch_data_high,
                    fs=fs, draw=False, method=method,
                    low_fq_range=low_fq_range,
                    low_fq_width=low_fq_width,
                    high_fq_range=high_fq_range,
                    high_fq_width=high_fq_width,
                    save_name=filename_full + save_name_suffix,
                    progress_bar=False)

                low_fq, high_fq, max_pac_value = get_maximum_pac(
                    comod, low_fq_range, high_fq_range)

                results.append((low_fq, high_fq, max_pac_value))
            bar.update_with_increment_value(1)

        results = np.array(results)
        results = np.reshape(results, (len(events), n_splits, n_epochs, 3))
        results = np.swapaxes(results, 1, 2)
        split_centers = [(start + stop) / 2. for start, stop in splits]

        fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        cmap_csm = plt.get_cmap('Blues')
        cmap_csp = plt.get_cmap('Reds')
        for cmap, results_by_event in zip([cmap_csm, cmap_csp], results):
            for i_epoch, results_by_epochs in enumerate(results_by_event):
                col = cmap(1.0 - i_epoch / n_epochs * 0.75)
                axs[0].plot(split_centers, results_by_epochs[:, 0], color=col)
                axs[1].plot(split_centers, results_by_epochs[:, 1], color=col)
                axs[2].plot(split_centers, results_by_epochs[:, 2], color=col)

        axs[0].set_title('Low frequency (Hz)')
        axs[1].set_title('High frequency (Hz)')
        axs[2].set_title('Maximum PAC value (%s)' % method)
        axs[2].set_xlabel("Time (sec) (window of %s sec)" % t_split)
        axs[0].set_ylim((low_fq_range[0], low_fq_range[-1]))
        axs[1].set_ylim((high_fq_range[0], high_fq_range[-1]))
        axs[2].set_ylim((0, axs[2].get_ylim()[1]))

        # add suptitle
        fig.subplots_adjust(top=0.85)
        name = '%s_ch(%s,%s)%s' % (raw_file_name, channel_low, channel_high,
                                   save_name_suffix)
        fig.suptitle('Maximum PAC, %s, driver low_fq_width = %.2f Hz\n'
                     'CS+ (reds) vs CS- (blues), darker = first epochs'
                     % (name, low_fq_width), fontsize=14)

        fig.savefig(name + '.png')
        print('Saved: %s' % (name + '.png'))
        plt.close('all')


if __name__ == '__main__':
    example_all_channels_maximum_pac()
