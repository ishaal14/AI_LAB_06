import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')
print(df.head())

data = df[['Income', 'Age']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_normalized)

centroids = kmeans.cluster_centers_
centroids_original_scale = scaler.inverse_transform(centroids)

print("Centroids (original scale):")
print(centroids_original_scale)

df['Cluster'] = kmeans.labels_
new_centroids = df.groupby('Cluster')[['Income', 'Age']].mean()

print("New Centroids (mean values of corresponding clusters):")
print(new_centroids)

plt.scatter(df['Income'], df['Age'], c=df['Cluster'], cmap='viridis')
plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Income')
plt.ylabel('Age')
plt.legend()
plt.show()