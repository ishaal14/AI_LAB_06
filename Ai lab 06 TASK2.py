import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Objects': ['08-1', '08-2', '08-3', '08-4', '08-5', '08-6', '08-7', '08-8'],
    'X': [1, 1, 1, 2, 1, 2, 1, 2, 4, 1, 1, 1, 2, 1, 2, 2, 1],
    'Y': [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2],
    'Z': [2, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

features = df[['X', 'Y', 'Z']]
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

centroids = kmeans.cluster_centers_
print("\nCluster Centroids (X, Y, Z):")
print(centroids)

print("\nCluster Assignments:")
print(df[['Objects', 'Cluster']])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X'], df['Y'], df['Z'], c=df['Cluster'], cmap='viridis', marker='o', s=100)
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200, label='Centroids')
ax.set_title("K-Means Clustering (K=2)")
