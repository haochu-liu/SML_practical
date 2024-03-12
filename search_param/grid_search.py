# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, zero_one_loss, make_scorer, classification_report


def grid_search(model, params, x, y, x_val, y_val, cv=5):
    """
    model: sklearn model
    param: parameter grid, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv: cross-validation splitting strategy
    x_val: X validation set
    y_val: y validation set
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    search = GridSearchCV(model, param_grid=params, cv=cv, scoring=scorer,
                          n_jobs=-1, verbose=10)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    # get the score for validation set
    val_model = model
    val_model.set_params(**best_param)
    val_model.fit(x, y)
    val_score = val_model.score(x_val, y_val)
    best_param['val_score'] = val_score

    return best_param


def rand_search(model, params, x, y, x_val, y_val, cv=5, n=20):
    """
    model: sklearn model
    param: parameter distributions, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv: cross-validation splitting strategy
    n: number of parameter settings that are sampled
    x_val: X validation set
    y_val: y validation set
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    search = RandomizedSearchCV(model, param_grid=params, cv=cv, scoring=scorer, n_iter=n,
                                n_jobs=-1, verbose=10, random_state=42)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    # get the score for validation set
    val_model = model
    val_model.set_params(**best_param)
    val_model.fit(x, y)
    val_score = val_model.score(x_val, y_val)
    best_param['val_score'] = val_score

    return best_param


def read_data():
    """
    read csv files
    """

    X_train_sc = pd.read_csv('data/X_train_sc.csv', header=None)
    X_val_sc = pd.read_csv('data/X_val_sc.csv', header=None)
    X_train_pca = pd.read_csv('data/X_train_pca.csv', header=None)
    X_val_pca = pd.read_csv('data/X_val_pca.csv', header=None)
    X_train_lda = pd.read_csv('data/X_train_lda.csv', header=None)
    X_val_lda = pd.read_csv('data/X_val_lda.csv', header=None)
    y_train = pd.read_csv('data/y_train.csv', header=None)
    y_val = pd.read_csv('data/y_val.csv', header=None)

    return X_train_sc, X_val_sc, X_train_pca, X_val_pca, X_train_lda, X_val_lda, y_train, y_val


