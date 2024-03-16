# Import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import time


# Load the training data and the test inputs
# Inputs of the training set
X_train = pd.read_csv('X_train.csv', index_col = 0, header=[0, 1, 2])
# Outputs of the training set
y_train = pd.read_csv('y_train.csv', index_col=0).squeeze('columns').to_numpy()
# Inputs of the test set
X_test = pd.read_csv('X_test.csv', index_col = 0, header=[0, 1, 2])

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit SVM and count our fitting time
svm = SVC(C=10, kernel='rbf', gamma='scale')
start_time = time.time()
svm.fit(X_train_scaled, y_train)
end_time = time.time()
print('Our training time is ', end_time - start_time)

# Make prediction from our SVM
y_pred = svm.predict(X_test_scaled)

# Export the predictions on the test data in csv format
prediction = pd.DataFrame(y_pred, columns=['Genre'])
prediction.index.name='Id'
prediction.to_csv('myprediction_try.csv') # export to csv file
print('The prediction file has been successfully saved!')


