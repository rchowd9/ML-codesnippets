# Import the necessary packages
# Your code here

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

heart = pd.read_csv("heart.csv")

# Input features: thalach and age
X = heart[["thalach", "age"]]

# Output feature: target
y = heart[["target"]]

# Create training and testing data with 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123
)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a support vector classifier with C=0.2 and a maximum of 500 iterations
SVC = LinearSVC(C=0.2, max_iter=500, random_state=123)
SVC.fit(X_train, np.ravel(y_train))
# Fit the support vector classifier according to the training data

# Evaluate model on testing data
score = SVC.score(X_test, np.ravel(y_test))
print(np.round(score, 3))

# Print the model weights
# w0
print("w0:", np.round(SVC.intercept_, 3))
# w1 and w2
print("w1 and w2:", np.round(SVC.coef_, 3))
