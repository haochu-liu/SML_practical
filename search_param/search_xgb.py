# Import libraries
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from grid_search import read_data, grid_search, rand_search, pipeline_search
from scipy.stats import randint


# Load data sets
X_train, X_val, y_train, y_val = read_data()

# Assuming y is your target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# set the grid
param_grid = {
    'classifier__max_depth': range(1,20),
    'classifier__learning_rate': np.arange(0.1,1,0.1),
    'classifier__subsample': np.arange(0.2,1,0.2),
    'classifier__colsample_bytree': np.arange(0.2,1,0.2),
    'classifier__n_estimators': [50, 100, 150, 200]
}

xgb = XGBClassifier(
    objective='multi:softmax',   # for multiclass classification
    num_class=8,                 # specify the number of classes
)
# get pipelines
scaler_pipe = pipeline_search(xgb)
pca_pipe = pipeline_search(xgb, process='PCA')
lda_pipe = pipeline_search(xgb, process='LDA')

# apply grid search
sc_best = rand_search(scaler_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)
pca_best = rand_search(pca_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)
lda_best = rand_search(lda_pipe, param_grid, X_train, y_train, X_val, y_val, n=10)

params = {'scaler': sc_best, 'pca': pca_best, 'lda': lda_best}
with open('search_param/xgb_param.json', 'w') as f:
    json.dump(params, f)


