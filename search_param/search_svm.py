# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.svm import SVC
from grid_search import read_data, grid_search, rand_search, pipeline_search


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# set the grid
param_grid = [{'classifier__C': [0.01, 0.1, 1, 10, 100],
               'classifier__kernel': ['poly'],
               'classifier__degree': [0, 1, 2, 3, 4, 5],
               'classifier__gamma': ['scale', 'auto']},
              {'classifier__C': [0.01, 0.1, 1, 10, 100],
               'classifier__kernel': ['rbf', 'sigmoid'],
               'classifier__gamma': ['scale', 'auto']}]

svm = SVC()
# get pipelines
scaler_pipe = pipeline_search(svm)
pca_pipe = pipeline_search(svm, process='PCA')
lda_pipe = pipeline_search(svm, process='LDA')

# apply grid search
sc_best = grid_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val)
pca_best = grid_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val)
lda_best = grid_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/svm_param.json', 'w') as f:
    json.dump(params, f)


