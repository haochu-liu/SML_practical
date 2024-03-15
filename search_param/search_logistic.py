# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from grid_search import read_data, grid_search, rand_search, pipeline_search


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# set the grid
param_grid = {'classifier__l1_ratio': [0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1],
              'classifier__C': [0.01, 0.1, 1, 10, 100]}

logistic = LogisticRegression(penalty='elasticnet', solver='saga')
# get pipelines
scaler_pipe = pipeline_search(logistic)
pca_pipe = pipeline_search(logistic, process='PCA')
lda_pipe = pipeline_search(logistic, process='LDA')

# apply grid search
sc_best = grid_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val)
pca_best = grid_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val)
lda_best = grid_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/logistic_param.json', 'w') as f:
    json.dump(params, f)


