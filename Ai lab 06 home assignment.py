import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Create the dataset
data = {
    "Example No.": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Color": ["Red", "Red", "Red", "Yellow", "Yellow", "Yellow", "Yellow", "Yellow", "Red", "Red"],
    "Type": ["Sports", "Sports", "Sports", "Sports", "Sports", "SUV", "SUV", "SUV", "SUV", "Sports"],
    "Origin": ["Domestic", "Domestic", "Domestic", "Domestic", "Imported", "Imported", "Imported", "Domestic", "Imported", "Imported"],
    "Stolen?": ["Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "Yes"]
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Preprocess the data: Convert categorical data to numerical using LabelEncoder
label_encoders = {}
for column in ["Color", "Type", "Origin", "Stolen?"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare the data for clustering (exclude "Example No.")
X = df.drop(columns=["Example No."])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

# Output clustered data
print("Clustered Data:")
print(df)

# Visualize the clusters using the first two features (Color and Type) for simplicity
plt.scatter(X["Color"], X["Type"], c=df["Cluster"], cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c="red", label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Color")
plt.ylabel("Type")
plt.legend()
plt.show(