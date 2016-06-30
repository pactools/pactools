import os
import mne

from pactools.utils.spliter import blend_and_ravel


def load_data_example():
    directory = '/data/tdupre/rat_data/LTM1_L1-L2/'
    if not os.path.exists(directory):
        directory = '/home/tom/data/LTM1-2/'

    filename = 'c2_e1_[0.0-30.0]-epo.fif'
    loaded_epochs = mne.read_epochs(directory + filename)

    fs = loaded_epochs.info['sfreq']
    sig = loaded_epochs.get_data()

    #remove the first t_evok seconds with the ERP
    t_evok = 1.0
    sig = sig[:, 0, int(t_evok * fs):]

    # concatenate the epochs into one signal, with a smooth transition of 0.1s
    sig = blend_and_ravel(sig, n_blend=int(0.1 * fs))

    return sig, fs
