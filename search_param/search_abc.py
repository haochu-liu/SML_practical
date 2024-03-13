# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import AdaBoostClassifier
from grid_search import read_data, grid_search, rand_search, pipeline_search
from scipy.stats import randint


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# set the grid
# rough search
# param_grid = {'n_estimators': list(range(1, 100, 5)),
#               'algorithm': ['SAMME', 'SAMME.R']}
# best n_estimators are 96, 81, 96

# reset the grid as randint(75, 100)
param_grid = {'classifier__n_estimators': randint(75, 125),
              'classifier__algorithm': ['SAMME', 'SAMME.R']}


abc = AdaBoostClassifier(random_state=42)
# get pipelines
scaler_pipe = pipeline_search(abc)
pca_pipe = pipeline_search(abc, process='PCA')
lda_pipe = pipeline_search(abc, process='LDA')

# apply grid search
sc_best = rand_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)
pca_best = rand_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)
lda_best = rand_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/abc_param.json', 'w') as f:
    json.dump(params, f)


