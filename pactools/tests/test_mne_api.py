import numpy as np
import matplotlib.pyplot as plt
import mne

from pactools.utils.testing import assert_equal, assert_true
from pactools.utils.testing import assert_array_equal
from pactools.comodulogram import Comodulogram
from pactools.mne_api import raw_to_mask, MaskIterator


def create_empty_raw():
    n_channels = 3
    n_points = 300
    data = np.empty([n_channels, n_points])
    fs = 200.
    info = mne.create_info(ch_names=n_channels, sfreq=fs)
    raw = mne.io.RawArray(data, info)
    return raw


def test_mask_iterator():
    # Test the mask iterator length and the shape of the elements
    fs = 200.
    tmin, tmax = -0.05, 0.2
    n_points = 200
    events = np.arange(0, n_points, int(fs * 0.3))
    masks = MaskIterator(events, tmin, tmax, n_points, fs)
    masks_list = [m for m in masks]

    # test that the length is correct
    n_masks_0 = len(masks)
    n_masks_1 = len(masks_list)
    assert_equal(n_masks_0, n_masks_1)

    # test that the masks are of the correct shape
    for mask in masks_list:
        assert_array_equal(mask.shape, (1, n_points))


def test_mask_iterator_length():
    # Test the mask iterator length
    rng = np.random.RandomState(0)
    fs = 200.
    n_points = 200

    tmin, tmax = -0.05, 0.2
    events = np.arange(0, n_points, int(fs * 0.3))
    masks = MaskIterator(events, tmin, tmax, n_points, fs)
    assert_equal(len(masks), 1)

    tmin, tmax = [-0.05, -0.02, 0], [-0.2, -0.2, 0.1]
    events = np.arange(0, n_points, int(fs * 0.3))
    masks = MaskIterator(events, tmin, tmax, n_points, fs)
    assert_equal(len(masks), len(tmin))

    n_kinds = 3
    tmin, tmax = -0.05, 0.2
    events = np.arange(0, n_points, 10)
    events = np.vstack([events, rng.randint(n_kinds, size=len(events))]).T
    masks = MaskIterator(events, tmin, tmax, n_points, fs)
    assert_equal(len(masks), n_kinds)


def test_raw_to_mask_pipeline():
    # Smoke test for the pipeline MNE.Raw -> raw_to_mask -> Comodulogram
    path = mne.datasets.sample.data_path()
    raw_fname = path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
    event_fname = path + ('/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif')
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    events = mne.read_events(event_fname)

    low_sig, high_sig, mask = raw_to_mask(raw, ixs=(8, 10), events=events,
                                          tmin=-1., tmax=5.)
    estimator = Comodulogram(
        fs=raw.info['sfreq'], low_fq_range=np.linspace(1, 5, 5),
        low_fq_width=2., method='tort', progress_bar=False)
    estimator.fit(low_sig, high_sig, mask)
    estimator.plot(tight_layout=False)
    plt.close('all')


def test_raw_to_mask_no_events():
    # test that no events gives masks entirely False
    raw = create_empty_raw()
    low_sig, high_sig, mask = raw_to_mask(raw, ixs=0, events=None,
                                          tmin=None, tmax=None)
    for one_mask in mask:
        assert_true(np.all(~one_mask))

    # test with no events and tmin, tmax
    tmin, tmax = 0.2, 0.6
    low_sig, high_sig, mask = raw_to_mask(raw, ixs=0, events=None,
                                          tmin=tmin, tmax=tmax)
    for one_mask in mask:
        assert_equal(one_mask[~one_mask].size,
                     int((tmax - tmin) * raw.info['sfreq']))
