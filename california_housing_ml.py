
# California Housing: Feature Engineering, PCA,
# K-Means, Linear Regression, Hierarchical Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score
)
from scipy.cluster.hierarchy import linkage, dendrogram

# 1. LOAD DATASET
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset statistics:")
print(df.describe())

# 2. FEATURE ENGINEERING
# Create new features from existing columns
df["RoomsPerPerson"] = (
    df["AveRooms"] / df["AveOccup"].replace(0, np.nan)
)

df["BedroomsPerRoom"] = (
    df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
)

# Separate input features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("\nNew features created:")
print(X_imputed.head())

# Feature correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    annot=False
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 3. STANDARDIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 4. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total variance retained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2),
    "%"
)

plt.figure(figsize=(8, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=5,
    alpha=0.4
)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - California Housing")
plt.tight_layout()
plt.show()

# 5. K-MEANS CLUSTERING
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans_labels = kmeans.fit_predict(X_scaled)

df["KMeans_Cluster"] = kmeans_labels

print("\nK-Means cluster counts:")
print(df["KMeans_Cluster"].value_counts().sort_index())

# Silhouette score on a sample to save time
rng = np.random.RandomState(42)
sample_indices = rng.choice(
    len(X_scaled),
    size=2000,
    replace=False
)

print(
    "K-Means silhouette score:",
    round(
        silhouette_score(
            X_scaled[sample_indices],
            kmeans_labels[sample_indices]
        ),
        3
    )
)

plt.figure(figsize=(8, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans_labels,
    cmap="viridis",
    s=5
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clusters Visualized Using PCA")
plt.tight_layout()
plt.show()

# 6. LINEAR REGRESSION
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42
)

# Fit the scaler using training data only
reg_scaler = StandardScaler()
X_train_scaled = reg_scaler.fit_transform(X_train)
X_test_scaled = reg_scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
print("R2 Score:", round(r2_score(y_test, y_pred), 3))

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.3, s=8)
plt.xlabel("Actual House Value ($100,000 units)")
plt.ylabel("Predicted House Value ($100,000 units)")
plt.title("Actual vs Predicted House Values")
plt.tight_layout()
plt.show()

# 7. HIERARCHICAL CLUSTERING
# Use a random sample because hierarchical clustering
# can be slow on the entire dataset.
hier_indices = rng.choice(
    len(X_scaled),
    size=1000,
    replace=False
)

X_hier = X_scaled[hier_indices]

hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

hier_labels = hierarchical.fit_predict(X_hier)

print("\nHierarchical cluster counts:")
print(pd.Series(hier_labels).value_counts().sort_index())

print(
    "Hierarchical silhouette score:",
    round(silhouette_score(X_hier, hier_labels), 3)
)

# Dendrogram
Z = linkage(X_hier[:100], method="ward")

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode="level", p=4)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample groups")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Cluster visualization using the first two PCA components
plt.figure(figsize=(8, 5))
plt.scatter(
    X_pca[hier_indices, 0],
    X_pca[hier_indices, 1],
    c=hier_labels,
    cmap="viridis",
    s=15
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clustering - Sample Data")
plt.tight_layout()
plt.show()

print("\nAll five tasks completed successfully!")
