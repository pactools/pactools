from __future__ import print_function
import os

import numpy as np
from mne import Epochs

from pactools.preprocess import fill_gap, preprocess
from pactools.io.smr2mne import sig2mne
from pactools.io.matfiles import sig2mat, mat2sig

RAW_FILE = '/data/tdupre/rat_data/LTM1_L1-L2.smr'  # data from rat experiment


def params_rat():
    draw = ''
    # Available plots are:
    #     'e' = extract_and_fill
    #     'c' = carrier filter
    #     'd' = dehumming
    #     'f' = electrical network frequency
    #     'w' = whitening
    #     'g' = fill the 50 Hz gap
    #     'z' = plot all above plots.

    preprocess_params = {
        # raw file
        'raw_file': RAW_FILE,
        'fs': 10000.,  # sampling frequency
        'start': None,  # in sec
        'stop': None,  # in sec

        # decimation
        'decimation_factor': 30,

        # dehumming
        'enf': 50.0,  # electrical network frequency
        'blklen': 2048,  # block length for dehumming

        # custom function
        'custom_func': fill_gap,  # fill the wide gap at 50 Hz

        # plot some figures
        'draw': draw,
    }

    extract_params = {
        # band-pass filter
        'fc_array': [3.0],  # carrier frequency
        'n_cycles': None,  # number of cycles in the band-pass filter
        'bandwidth': 1.0,  # bandwidth (Hz) of the band-pass filter
        # (use bandwidth OR n_cycles)

        # filling strategy
        'fill': 4,  # filling method {0, 1, 2, 3, 4}

        # whitening
        'whitening': 'after',  # None, 'before', 'after'
        'ordar': 37,  # ordar of the whitening AR filter
        'enf': 50.0,  # electrical network frequency (default: 50.0)

        # normalization to unit variance
        'normalize': True,  # normalize the variance of signals

        # plot some figures
        'draw': draw,
    }
    return preprocess_params, extract_params


def get_save_directory(raw_file):
    # decompose the raw filename
    dirname = os.path.dirname(raw_file)
    filename, extension = os.path.splitext(os.path.basename(raw_file))

    # create directory
    save_dir = os.path.join(dirname, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def example_preprocessing_channel_by_channel():
    """
    Example of preprocessing of a raw file
    """
    # load parameters and preprocess the data
    # (load raw file, decimate, dehumm, fill the 50 Hz gap)
    preprocess_params, _ = params_rat()
    raw_file = preprocess_params['raw_file']
    save_dir = get_save_directory(raw_file)
    save_name = '%s/dehummed.mat' % save_dir

    # skip preprocessing and load if the file exists
    if os.path.isfile(save_name):
        data = mat2sig(save_name)
        sigs, fs, events = data['sigs'], data['fs'], data['events']
    else:
        sigs, fs, events = preprocess(**preprocess_params)
        sig2mat(save_name, sigs=sigs, fs=fs, events=events)

    # preprocess through MNE to remove bad epochs, select channels, ...
    mne_raw, mne_events = sig2mne(sigs, fs, events)

    # select the channel
    for channel in range(8):
        # select CS+ or CS- events
        for event_id in [-1, +1]:
            # select time around the events
            for tmin, tmax in [(-30., 0), (0., 30), (30., 60)]:
                # save name
                raw_file = preprocess_params['raw_file']
                save_dir = get_save_directory(raw_file)

                save_name = ('%s/c%d_e%d_[%.1f-%.1f]-epo.fif'
                             % (save_dir, channel, event_id, tmin, tmax))

                # if the file already exists, we skip it
                if not os.path.isfile(save_name):
                    picks = [channel]   # select channel
                    epochs = Epochs(mne_raw, mne_events,
                                    event_id=event_id,
                                    preload=True, picks=picks,
                                    tmin=tmin, tmax=tmax,
                                    baseline=None)
                    picks = [0]
                    # plot and reject bad epochs
                    epochs.plot(picks=picks, scalings={'misc': 0.2},
                                n_channels=4, n_epochs=3, block=True)

                    # save the epochs object (bad epochs are not saved)
                    if epochs.events.size > 0:
                        epochs.save(save_name)
                        print('Saved: %s' % save_name)


def example_preprocessing_all_channels():
    """
    Example of preprocessing of a raw file
    """
    # load parameters and preprocess the data
    # (load raw file, decimate, dehumm, fill the 50 Hz gap)
    preprocess_params, _ = params_rat()
    raw_file = preprocess_params['raw_file']
    save_dir = get_save_directory(raw_file)
    save_name = '%s/dehummed.mat' % save_dir

    # skip preprocessing and load if the file exists
    if os.path.isfile(save_name):
        data = mat2sig(save_name)
        sigs, fs, events = data['sigs'], data['fs'], data['events']
    else:
        sigs, fs, events = preprocess(**preprocess_params)
        sig2mat(save_name, sigs=sigs, fs=fs, events=events)

    # preprocess through MNE to remove bad epochs, select channels, ...
    mne_raw, mne_events = sig2mne(sigs, fs, events)

    # select CS+ or CS- events
    for event_id in [-1, +1]:
        # select time around the events
        tmin = -30.0
        tmax = 60.0

        # save name
        raw_file = preprocess_params['raw_file']
        save_dir = get_save_directory(raw_file)

        save_name = ('%s/e%d_[%.1f-%.1f]-epo.fif'
                     % (save_dir, event_id, tmin, tmax))

        # if the file already exists, we skip it
        if not os.path.isfile(save_name):
            picks = np.arange(8)   # all channels
            epochs = Epochs(mne_raw, mne_events,
                            event_id=event_id,
                            preload=True, picks=picks,
                            tmin=tmin, tmax=tmax,
                            baseline=None)

            # plot and reject bad epochs
            epochs.plot(picks=picks, scalings={'misc': 1.0},
                        n_channels=8, n_epochs=3, block=True)

            # save the epochs object (bad epochs are not saved)
            if epochs.events.size > 0:
                epochs.save(save_name)
                print('Saved: %s' % save_name)

if __name__ == '__main__':
    example_preprocessing_all_channels()
