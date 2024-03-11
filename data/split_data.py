# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# Load the training data and the test inputs
X_train = pd.read_csv('X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train = pd.read_csv('y_train.csv', index_col=0).squeeze('columns').to_numpy() # outputs of the training set

# Split the data into training and test sets
split_prop = 0.2
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=split_prop, random_state=42)

# Scale the features using StandardScaler
scalerx = StandardScaler()
X_train = scalerx.fit_transform(X_train)
X_val = scalerx.transform(X_val)

# Save the scaled data sets
np.savetxt('data/X_train_sc.csv', X_train, delimiter=',')
np.savetxt('data/X_val_sc.csv', X_val, delimiter=',')


