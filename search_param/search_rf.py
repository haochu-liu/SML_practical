# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from grid_search import read_data, grid_search, rand_search


# Load data sets
X_train_sc, X_val_sc, X_train_pca, X_val_pca, X_train_lda, X_val_lda, y_train, y_val = read_data()
y_train = y_train[0].values
y_val = y_val[0].values

# apply grid search
param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
              'max_depth': [3, 5, 10, 15, 20, 30],
              'n_estimators': [10, 20, 50, 100, 200]}

rfc = RandomForestClassifier()
sc_best = grid_search(rfc, param_grid, X_train_sc, y_train, X_val_sc, y_val)
pca_best = grid_search(rfc, param_grid, X_train_pca, y_train, X_val_pca, y_val)
lda_best = grid_search(rfc, param_grid, X_train_lda, y_train, X_val_lda, y_val)

sc_pca_lda = [sc_best, pca_best, lda_best]
with open('search_param/rf_param.json', 'w') as f:
    json.dump(sc_pca_lda, f)


