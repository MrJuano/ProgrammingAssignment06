import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Load the full grocery customer dataset and take a random sample of 500 instances
data = pd.read_csv('customer_personality.csv').sample(500, random_state=123) # your code here , random_state=123)

# Use StandardScaler() to standardize input features
X = data[['Fruits', 'Meats']]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

# Apply DBSCAN with epsilon=1 and min_samples = 8   
dbscan = DBSCAN(eps=1, min_samples=8) # Your code here 
dbscan = dbscan.fit(X)

# Print the cluster labels and core point indices
print('Labels:', dbscan.labels_) # Your code here )
print('Core points:', dbscan.core_sample_indices_) # Your code here ) 
print('Number of core points:', len(dbscan.core_sample_indices_)) # Your code here )

# Add the cluster labels to the dataset as strings
data['clusters'] = dbscan.labels_.astype(str)

# Sort by cluster label (for plotting purposes)
data.sort_values(by='clusters', inplace=True)

# Plot clusters on the original data
p = sns.scatterplot(data=data, x='Fruits',
                    y='Meats', hue='clusters',
                    style='clusters')
p.set_xlabel('Fruits', fontsize=16)
p.set_ylabel('Meats', fontsize=16)
p.legend(title='DBSCAN')
