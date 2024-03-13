# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from grid_search import read_data, grid_search, rand_search, pipeline_search


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# set the grid
param_grid = {'classifier__criterion': ['gini', 'entropy', 'log_loss'],
              'classifier__max_depth': [3, 5, 10, 15, 20, 30],
              'classifier__n_estimators': [10, 20, 50, 100, 200]}

rfc = RandomForestClassifier()
# get pipelines
scaler_pipe = pipeline_search(rfc)
pca_pipe = pipeline_search(rfc, process='PCA')
lda_pipe = pipeline_search(rfc, process='LDA')

# apply grid search
sc_best = grid_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val)
pca_best = grid_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val)
lda_best = grid_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/rf_param.json', 'w') as f:
    json.dump(params, f)


