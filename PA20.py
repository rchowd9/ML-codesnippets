import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("mpg_clean.csv")

# Create a dataframe X containing the input features
X = df.drop(columns=["name", "origin"])
# Create a dataframe y containing the output feature origin
y = df[["origin"]]

# Get user-input n_estimators and max_features
estimators = int(input())
max_features = int(input())

# Initialize and fit a random forest classifier with user-input number of decision trees, 
# user-input number of features considered at each split, and a random state of 123

rfModel = RandomForestClassifier(n_estimators=estimators, max_features=max_features, random_state=123)
rfModel.fit(X, np.ravel(y))

# Calculate prediction accuracy
score = rfModel.score(X, y)
print(round(score, 4))

# Calculate the permutation importance using the default parameters and a random state of 123
result = permutation_importance(rfModel, X, y, random_state=123)

# Variable importance table
importance_table = pd.DataFrame(
    data={"feature": rfModel.feature_names_in_,"permutation importance": result.importances_mean}
).sort_values("permutation importance", ascending=False)

print(importance_table)
