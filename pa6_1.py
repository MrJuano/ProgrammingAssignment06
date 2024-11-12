# Import needed packages
# Your code here
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

healthy = pd.read_csv('healthy_lifestyle.csv')

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[['sunshine_hours', 'happiness_levels']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])
X = X.dropna()

# Initialize a k-means clustering algorithm with a user-defined number of clusters, init='random', n_init=10, 
# random_state=123, and algorithm='elkan'
# Your code here
kmeans = KMeans(n_clusters=number, init='random', n_init=10, random_state=123, algorithm='elkan')

# Fit the algorithm to the input features
# Your code here
kmeans.fit(X)

# Find and print the cluster centroids
centroid = kmeans.cluster_centers_ # Your code here
print("Centroids:", np.round(centroid,4))

# Find and print the cluster inertia
inertia = kmeans.inertia_ # Your code here
print("Inertia:", np.round(inertia,4))
