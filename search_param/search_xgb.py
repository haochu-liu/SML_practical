# Import libraries
import pandas as pd
import numpy as np
import json
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from grid_search import read_data, grid_search, rand_search
from scipy.stats import randint


# Load data sets
X_train_sc, X_val_sc, X_train_pca, X_val_pca, X_train_lda, X_val_lda, y_train, y_val = read_data()
y_train = y_train[0].values
y_val = y_val[0].values

# Assuming y is your target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_val = label_encoder.transform(y_val)

# apply grid search
param_grid = {
    'max_depth': range(1,20),
    'learning_rate': np.arange(0.1,1,0.1),
    'subsample': np.arange(0.2,1,0.2),
    'colsample_bytree': np.arange(0.2,1,0.2),
    'n_estimators': [50, 100, 150, 200]
}

xgb = XGBClassifier(
    objective='multi:softmax',   # for multiclass classification
    num_class=8,                 # specify the number of classes
)
sc_best = rand_search(xgb, param_grid, X_train_sc, y_train, X_val_sc, y_val, n=10)
pca_best = rand_search(xgb, param_grid, X_train_pca, y_train, X_val_pca, y_val, n=10)
lda_best = rand_search(xgb, param_grid, X_train_lda, y_train, X_val_lda, y_val, n=10)

sc_pca_lda = [sc_best, pca_best, lda_best]
with open('search_param/xgb_param.json', 'w') as f:
    json.dump(sc_pca_lda, f)


