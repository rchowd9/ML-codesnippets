# Import needed packages
# Your code here
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


healthy = pd.read_csv("healthy_lifestyle.csv")

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[["sunshine_hours", "happiness_levels"]]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=["sunshine_hours", "happiness_levels"])
X = X.dropna()

# Initialize a k-means clustering model named kmeansHealthy with a user-defined number of clusters, 
# init="random", n_init=10, random_state=123, and algorithm="elkan"
# Your code here
kmeansHealthy = KMeans(n_clusters=number, init="random", random_state=123, n_init=10, algorithm="elkan")
kmeansHealthy.fit(X)

# Fit the algorithm to the input features
# Your code here

# Find and print the cluster centroids
centroid = kmeansHealthy.cluster_centers_
print("Centroids:", np.round(centroid,4))

# Find and print the cluster inertia
inertia = kmeansHealthy.inertia_
print("Inertia:", np.round(inertia,4))
