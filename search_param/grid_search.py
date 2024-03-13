# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import accuracy_score, zero_one_loss, make_scorer, classification_report


def pipeline_search(model, process='Scaler'):
    """
    model: sklearn classifier
    process: Scaler / PCA / LDA
    """

    if process == 'Scaler':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        return pipeline
    elif process == 'PCA':
        p_PCA = 25 # from notebook pictures
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=p_PCA, random_state=42)),
            ('classifier', model)
        ])
        return pipeline
    elif process == 'LDA':
        pipeline = Pipeline([
            ('lda', LinearDiscriminantAnalysis(n_components=None)),
            ('classifier', model)
        ])
        return pipeline
    else:
        print('Unvalid process input')


def grid_search(estimator, params, x, y, x_val, y_val, cv_n=5):
    """
    estimator: sklearn pipeline
    param: parameter grid, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv_n: number of folds in cross-validation
    x_val: X validation set
    y_val: y validation set
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    cv = KFold(n_splits=cv_n, shuffle=True, random_state=42)
    search = GridSearchCV(estimator, param_grid=params, cv=cv, scoring=scorer,
                          n_jobs=-1, verbose=10)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    # get the score for validation set
    val_model = search.best_estimator_
    val_score = val_model.fit(x, y).score(x_val, y_val)
    best_param['val_score'] = val_score

    return best_param


def rand_search(estimator, params, x, y, x_val, y_val, cv_n=5, n=20):
    """
    estimator: sklearn pipeline
    param: parameter distributions, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv_n: number of folds in cross-validation
    n: number of parameter settings that are sampled
    x_val: X validation set
    y_val: y validation set
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    cv = KFold(n_splits=cv_n, shuffle=True, random_state=42)
    search = RandomizedSearchCV(estimator, param_distributions=params, cv=cv, scoring=scorer, n_iter=n,
                                n_jobs=-1, verbose=10, random_state=42)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    # get the score for validation set
    val_model = search.best_estimator_
    val_score = val_model.fit(x, y).score(x_val, y_val)
    best_param['val_score'] = val_score

    return best_param


def read_data():
    """
    read csv files
    """

    X_train = pd.read_csv('data/X_train.csv', header=None)
    X_val = pd.read_csv('data/X_val.csv', header=None)
    y_train = pd.read_csv('data/y_train.csv', header=None)
    y_train = y_train[0].values
    y_val = pd.read_csv('data/y_val.csv', header=None)
    y_val = y_val[0].values

    return X_train, X_val, y_train, y_val


