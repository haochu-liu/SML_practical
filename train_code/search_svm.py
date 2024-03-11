# Import libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC


# Load data sets
X_train_sc = pd.read_csv('data/X_train_sc.csv')


#
param_grid = [{'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['poly'], 'degree': [0, 1, 2, 3, 4, 5], 'gamma': ['scale', 'auto']},
              {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'sigmoid'], 'gamma': ['scale', 'auto']}]