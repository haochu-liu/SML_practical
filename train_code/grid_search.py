# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, zero_one_loss, make_scorer, classification_report


def grid_search(model, params, x, y, cv=5):
    """
    model: sklearn model
    param: parameter grid, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv: cross-validation splitting strategy
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    search = GridSearchCV(model, param_grid=params, cv=cv, scoring=scorer,
                          n_jobs=-1, verbose=10, random_state=42)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    return(best_param)


def rand_search(model, params, x, y, cv=5, n=20):
    """
    model: sklearn model
    param: parameter distributions, dictionary or a list of dictionaries
    x: X train set
    y: y train set
    cv: cross-validation splitting strategy
    n: number of parameter settings that are sampled
    """

    # fit grid search CV
    scorer = make_scorer(zero_one_loss, greater_is_better=False)
    search = RandomizedSearchCV(model, param_grid=params, cv=cv, scoring=scorer, n_iter=n,
                                n_jobs=-1, verbose=10)
    search.fit(x, y)

    # get best parameters
    best_param = search.best_params_
    print('The best parameters are\n', best_param)

    return(best_param)


