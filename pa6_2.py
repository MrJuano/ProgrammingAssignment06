import pandas as pd
import seaborn as sns
import numpy as np

# Import needed sklearn packages
# Your code here
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Import needed scipy packages
# Your code here
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage

# Silence warning
import warnings
warnings.filterwarnings('ignore')


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

# Initialize and fit an agglomerative clustering model using ward linkage in scikit-learn, with a user-defined
# number of clusters
# Your code here
agg_clustering = AgglomerativeClustering(n_clusters=number, linkage='ward')

# Add cluster labels to input feature dataframe
X['labels']= agg_clustering.fit_predict(X)# Your code here
print(X.head())

# Perform agglomerative clustering using SciPy

# Calculate the distances between all instances
# Your code here
distances = pdist(X[['sunshine_hours', 'happiness_levels']])

# Convert the distance matrix to a square matrix
# Your code here
square_distances = squareform(distances)

# Define a clustering model with ward linkage
clustersHealthyScipy = linkage(distances, method='ward') # Your code here

print('First five rows of the linkage matrix from SciPy:\n', np.round(clustersHealthyScipy[:5, :], 0))
