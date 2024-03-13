from search_param.grid_search import read_data, decode_dict
from sklearn.metrics import accuracy_score
import json
import time


def fit_model(model, path, param_key, X_train, y_train, X_val, y_val):
    """
    model: sklearn model
    path: path of the json file of parameters
    param_key: 'scaler' / 'pca' / 'lda'
    X_train, y_train, X_val, y_val: train set and validation set
    """

    # Load JSON file into Python dictionary
    with open(path, 'r') as f:
        param = json.load(f)

    param[param_key].popitem()
    param = decode_dict(param[param_key])

    # Set the parameters
    model.set_params(**param)

    # Count the time taken for training the model
    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()
    training_time  = end_time - start_time

    # Get train accuracy and predicted y
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Get test accuracy and predicted y
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)

    return training_time, train_acc, test_acc, y_train_pred, y_val_pred


