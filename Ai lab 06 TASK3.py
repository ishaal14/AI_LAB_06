import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = {
'Objects': ['OB-1', 'OB-2', 'OB-3', 'OB-4', 'OB-5', 'OB-6', 'OB-7', 'OB-8'],
'X': [1, 1, 1, 2, 1, 2, 1, 2],
'Y': [4, 2, 4, 1, 1, 4, 1, 1],
'Z': [1, 2, 1, 2, 1, 2, 2, 1]
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
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.legend()
plt.show()