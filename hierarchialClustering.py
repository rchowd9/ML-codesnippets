import pandas as pd
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


# Import needed sklearn packages
# Your code here

# Import needed scipy packages
# Your code here

# Silence warning
import warnings
warnings.filterwarnings("ignore")


healthy = pd.read_csv("healthy_lifestyle.csv")
healthy = healthy.dropna(subset = ["sunshine_hours", "happiness_levels"])

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[["sunshine_hours", "happiness_levels"]]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=["sunshine_hours", "happiness_levels"])

# Initialize and fit an agglomerative clustering model using ward linkage in scikit-learn, with a user-defined
# number of clusters
# Your code here
clusters = AgglomerativeClustering(n_clusters=number, linkage="ward")
clusters.fit(X)


# Add cluster labels to the healthy dataframe
healthy["labels"]= clusters.labels_
print(healthy[["sunshine_hours", "happiness_levels", "labels"]].head())

# Calculate the distances between all instances
# Your code here
distances = pdist(X)

# Convert the distance matrix to a square matrix
# Your code here
dist_matrix = squareform(distances)

# Define a clustering model with ward linkage
clustersHealthyScipy = linkage(dist_matrix, method="ward")

print("First five rows of the linkage matrix from SciPy:\n", np.round(clustersHealthyScipy[:5, :], 0))
