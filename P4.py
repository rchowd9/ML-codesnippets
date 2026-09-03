# Import the necessary modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read in the file fg_attempt.csv
field_goal = pd.read_csv("fg_attempt.csv")

# Create a dataframe X containing "Distance" and "ScoreDiffPreKick"
X = field_goal[['Distance', 'ScoreDiffPreKick']]# Your code here 
# Create a dataframe y containing "Outcome"
y = field_goal[['Outcome']]# Your code here

# Flatten y into an array
# Your code here
yArray = np.ravel(y)

# Initialize a LogisticRegression() model
logisticModel = LogisticRegression()

# Fit the model
# Your code here
logisticModel.fit(X, yArray)

# Input feature values for a sample instance
Distance = float(input())
ScoreDiffPreKick = float(input())

# Create a new dataframe with user-input Distance and ScoreDiffPreKick
XNew = pd.DataFrame({
    'Distance': [Distance], 
    'ScoreDiffPreKick': [ScoreDiffPreKick]
})

# Predict the outcome from the new data
pred = logisticModel.predict(XNew)  # Your code here
print(pred)

# Determine the accuracy of the model logisticModel
score = logisticModel.score(X, yArray) # Your code here
print(score)
