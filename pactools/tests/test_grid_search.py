import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from pactools import simulate_pac
from pactools.dar_model import extract_driver
from pactools.grid_search import DARSklearn, AddDriverDelay, MultipleArray
from pactools.grid_search import ExtractDriver, GridSearchCVProgressBar

fs = 200.  # Hz
high_fq = 50.0  # Hz
low_fq = 5.0  # Hz
low_fq_width = 1.0  # Hz

n_epochs = 3
n_points = 10000
noise_level = 0.8
t_plot = 2.0  # sec

raw_signal = np.array([
    simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                 low_fq_width=low_fq_width, noise_level=noise_level,
                 random_state=i) for i in range(n_epochs)
])

sigdriv, sigin, sigdriv_imag = extract_driver(
    sigs=raw_signal, fs=fs, low_fq=low_fq, bandwidth=low_fq_width,
    extract_complex=True, random_state=0, fill=2)


def test_ordar_ordriv_selection():
    # Test ordar and ordriv selection over grid-search and cross_validation
    param_grid = {'ordar': np.arange(0, 110, 10), 'ordriv': [0, 1]}
    model = DARSklearn(fs=fs, max_ordar=max(param_grid['ordar']))
    gscv = GridSearchCV(model, param_grid=param_grid, cv=3,
                        return_train_score=False)

    X = MultipleArray(sigin, sigdriv, sigdriv_imag)
    gscv.fit(X)

    assert gscv.best_params_['ordar'] < max(param_grid['ordar'])
    assert gscv.best_params_['ordriv'] > min(param_grid['ordar'])


def test_delay_selection():
    # Test delay selection over grid-search and cross_validation
    param_grid = {'add__delay': [-20, 0, 20]}
    model = Pipeline(steps=[
        ('add', AddDriverDelay()),
        ('dar', DARSklearn(fs=fs, ordar=10, ordriv=1, max_ordar=10)),
    ])
    gscv = GridSearchCV(model, param_grid=param_grid, cv=3,
                        return_train_score=False)

    X = MultipleArray(sigin, sigdriv, sigdriv_imag)
    gscv.fit(X)

    assert gscv.best_params_['add__delay'] == 0


def test_frequency_selection():
    # Test frequency selection over grid-search and cross_validation
    model = Pipeline(steps=[
        ('driver', ExtractDriver(fs=fs, low_fq=4., max_low_fq=7.,
                                 low_fq_width=low_fq_width, random_state=0)),
        ('add', AddDriverDelay()),
        ('dar', DARSklearn(fs=fs, ordar=20, ordriv=1, max_ordar=20)),
    ])

    param_grid = {
        'driver__low_fq': [3., 5., 7.],
        'driver__low_fq_width': [0.25, 0.5, 1.],
    }

    gscv = GridSearchCVProgressBar(model, param_grid=param_grid, cv=3,
                                   return_train_score=False, verbose=1)
    X = MultipleArray(raw_signal, None)
    gscv.fit(X)

    assert gscv.best_params_['driver__low_fq'] == 5
    assert gscv.best_params_['driver__low_fq_width'] == 1
