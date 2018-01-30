"""
Use grid-search and cross-validation
------------------------------------
This example creates an artificial signal with phase-amplitude coupling (PAC),
fits a DAR model over a grid-search of parameter, using cross_validation.

Cross-validation is done over epochs, with any strategy provided in
scikit-learn:
http://scikit-learn.org/stable/modules/classes.html#splitter-classes

Note that the score computed by a DARSklearn model is the log-likelihood from
which we subtracted the log-likelihood of an autoregressive model at order 0.
This is done to have a reference stable over cross-validation splits.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from pactools import simulate_pac
from pactools.dar_model import extract_driver
from pactools.grid_search import DARSklearn, MultipleArray

###############################################################################
# Let's first create an artificial signal with PAC.

fs = 200.  # Hz
high_fq = 50.0  # Hz
low_fq = 5.0  # Hz
low_fq_width = 1.0  # Hz

n_epochs = 3
n_points = 10000
noise_level = 0.4
t_plot = 2.0  # sec

signal = np.array([
    simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                 low_fq_width=low_fq_width, noise_level=noise_level,
                 random_state=i) for i in range(n_epochs)
])

###############################################################################
# Extract a low-frequency band, and fit a DAR model.

# Extract a low frequency band
sigdriv, sigin, sigdriv_imag = extract_driver(
    sigs=signal, fs=fs, low_fq=low_fq, bandwidth=low_fq_width,
    extract_complex=True, random_state=0, fill=2)

# Define the grid of parameters tested
param_grid = {'ordar': np.arange(0, 110, 10), 'ordriv': [0, 1, 2, 3]}

# Create a DAR model
model = DARSklearn(fs=fs, max_ordar=max(param_grid['ordar']))

# Plug the model and the parameter grid into a GridSearchCV estimator
gscv = GridSearchCV(model, param_grid=param_grid, return_train_score=False)

# Fit the grid-search
X = MultipleArray(sigin, sigdriv, sigdriv_imag)
gscv.fit(X)

###############################################################################
# Print the results of the grid search.

print("Best parameters set found over cross-validation:\n")
print(gscv.best_params_)

print("\nGrid scores (higher is better):\n")
for mean, std, params in zip(gscv.cv_results_['mean_test_score'],
                             gscv.cv_results_['std_test_score'],
                             gscv.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))

###############################################################################
# Plot the results of the grid search.

plt.figure()
ax = plt.gca()

df = pd.DataFrame(gscv.cv_results_)
index = 'param_ordar'
columns = [c for c in df.columns if c[:6] == 'param_']
columns.remove(index)
table_mean = df.pivot_table(index=index, columns=columns,
                            values=['mean_test_score'])
table_std = df.pivot_table(index=index, columns=columns,
                           values=['std_test_score'])
for col_mean, col_std in zip(table_mean.columns, table_std.columns):
    table_mean[col_mean].plot(ax=ax, yerr=table_std[col_std], marker='o',
                              label=col_mean)
plt.title('Grid-search results (higher is better)')
plt.ylabel('log-likelihood compared to an AR(0)')
plt.legend(title=table_mean.columns.names)
plt.show()
