# Import libraries
import pandas as pd
import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
from grid_search import read_data, grid_search, rand_search, pipeline_search


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# set the grid
param_grid = {'classifier__n_neighbors': list(range(1, 50)),
              'classifier__metric': ['minkowski', 'cosine', 'manhattan']}

knn = KNeighborsClassifier()
# get pipelines
scaler_pipe = pipeline_search(knn)
pca_pipe = pipeline_search(knn, process='PCA')
lda_pipe = pipeline_search(knn, process='LDA')

# apply grid search
sc_best = grid_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val)
pca_best = grid_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val)
lda_best = grid_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/knn_param.json', 'w') as f:
    json.dump(params, f)


