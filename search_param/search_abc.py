# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import AdaBoostClassifier
from grid_search import read_data, grid_search, rand_search
from scipy.stats import randint


# Load data sets
X_train_sc, X_val_sc, X_train_pca, X_val_pca, X_train_lda, X_val_lda, y_train, y_val = read_data()
y_train = y_train[0].values
y_val = y_val[0].values

# apply grid search
# rough search
# param_grid = {'n_estimators': list(range(1, 100, 5)),
#               'algorithm': ['SAMME', 'SAMME.R']}
# best n_estimators are 96, 81, 96

# reset the grid as randint(75, 100)
param_grid = {'n_estimators': randint(75, 125),
              'algorithm': ['SAMME', 'SAMME.R']}


abc = AdaBoostClassifier(random_state=42)
sc_best = rand_search(abc, param_grid, X_train_sc, y_train, X_val_sc, y_val, n=10)
pca_best = rand_search(abc, param_grid, X_train_pca, y_train, X_val_pca, y_val, n=10)
lda_best = rand_search(abc, param_grid, X_train_lda, y_train, X_val_lda, y_val, n=10)

sc_pca_lda = [sc_best, pca_best, lda_best]
with open('search_param/abc_param.json', 'w') as f:
    json.dump(sc_pca_lda, f)


