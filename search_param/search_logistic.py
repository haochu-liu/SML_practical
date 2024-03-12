# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from grid_search import read_data, grid_search, rand_search


# Load data sets
X_train_sc, X_val_sc, X_train_pca, X_val_pca, X_train_lda, X_val_lda, y_train, y_val = read_data()
y_train = y_train[0].values
y_val = y_val[0].values

# apply grid search
param_grid = {'l1_ratio': [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 1],
               'C': [0.01, 0.1, 1, 10, 100]}

svm = LogisticRegression()
sc_best = grid_search(svm, param_grid, X_train_sc, y_train, X_val_sc, y_val)
pca_best = grid_search(svm, param_grid, X_train_pca, y_train, X_val_pca, y_val)
lda_best = grid_search(svm, param_grid, X_train_lda, y_train, X_val_lda, y_val)

sc_pca_lda = [sc_best, pca_best, lda_best]
with open('search_param/logistic_param.json', 'w') as f:
    json.dump(sc_pca_lda, f)


