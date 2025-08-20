import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Create DataFrame
df = pd.read_csv("20-08\\es1\\dataset\\rsc\\Mall_Customers.csv")
print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Statistics:")
print(df.describe())


# Prepare data for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "="*60)
print("1. K-MEANS CLUSTERING ANALYSIS")
print("="*60)

# Find optimal number of clusters using Elbow Method
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot Elbow Method
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.grid(True)

# Choose optimal k (let's use k=5 based on typical Mall Customers analysis)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataframe
df['Cluster'] = cluster_labels

# 2. VISUALIZE CLUSTERS IN 2D PLOT
plt.subplot(1, 3, 3)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], 
               cluster_data['Spending Score (1-100)'], 
               c=colors[i], label=f'Cluster {i}', alpha=0.7, s=50)

# Plot centroids
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_original[:, 0], centroids_original[:, 1], 
           c='black', marker='x', s=200, linewidths=3, label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Clusters (K-Means)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. INTERPRET THE CLUSTERS
print("\n" + "="*60)
print("2. CLUSTER INTERPRETATION")
print("="*60)

cluster_summary = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'std', 'count'],
    'Spending Score (1-100)': ['mean', 'std'],
    'Age': ['mean', 'std']
}).round(2)

print("Cluster Summary Statistics:")
print(cluster_summary)

# Identify high-potential customers
print("\n" + "="*40)
print("3. HIGH-POTENTIAL CUSTOMERS ANALYSIS")
print("="*40)

cluster_characteristics = {}
for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    cluster_size = len(cluster_data)
    
    cluster_characteristics[i] = {
        'avg_income': avg_income,
        'avg_spending': avg_spending,
        'size': cluster_size,
        'potential': 'High' if avg_income > 60 and avg_spending > 60 else 
                    'Medium' if avg_income > 60 or avg_spending > 60 else 'Low'
    }
    
    print(f"\nCluster {i}:")
    print(f"  - Size: {cluster_size} customers")
    print(f"  - Average Income: ${avg_income:.1f}k")
    print(f"  - Average Spending Score: {avg_spending:.1f}")
    print(f"  - Potential Level: {cluster_characteristics[i]['potential']}")
    
    if avg_income > 60 and avg_spending > 60:
        print(f"  -  HIGH-VALUE SEGMENT: High income + High spending")
    elif avg_income > 60 and avg_spending < 40:
        print(f"  -  POTENTIAL SEGMENT: High income but low spending (opportunity!)")
    elif avg_income < 40 and avg_spending > 60:
        print(f"  -  LOYAL SEGMENT: Low income but high spending")
    else:
        print(f"  -  STANDARD SEGMENT: Moderate income and spending")

# 4. CALCULATE AVERAGE DISTANCE FROM CENTROIDS (CLUSTER COMPACTNESS)
print("\n" + "="*60)
print("4. CLUSTER COMPACTNESS ANALYSIS")
print("="*60)

for i in range(optimal_k):
    cluster_points = X_scaled[cluster_labels == i]
    centroid = kmeans.cluster_centers_[i]
    
    # Calculate distances from centroid
    distances = np.sqrt(np.sum((cluster_points - centroid)**2, axis=1))
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    print(f"\nCluster {i}:")
    print(f"  - Average distance from centroid: {avg_distance:.3f}")
    print(f"  - Standard deviation of distances: {std_distance:.3f}")
    print(f"  - Compactness level: {'High' if avg_distance < 0.8 else 'Medium' if avg_distance < 1.2 else 'Low'}")


# Final Summary
print("\n" + "="*60)
print("FINAL SUMMARY AND RECOMMENDATIONS")
print("="*60)

high_potential_clusters = [k for k, v in cluster_characteristics.items() if v['potential'] == 'High']
if high_potential_clusters:
    total_high_potential = sum(cluster_characteristics[k]['size'] for k in high_potential_clusters)
    print(f"\n HIGH-POTENTIAL CUSTOMERS IDENTIFIED:")
    print(f"   - Clusters: {high_potential_clusters}")
    print(f"   - Total customers: {total_high_potential}")
    print(f"   - Percentage of total: {total_high_potential/len(df)*100:.1f}%")

print(f"\n CLUSTERING PERFORMANCE:")
print(f"   - K-Means Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.3f}")


print(f"\n BUSINESS RECOMMENDATIONS:")
print(f"   1. Focus marketing efforts on high-income, high-spending clusters")
print(f"   2. Create retention strategies for loyal customers (low income, high spending)")
print(f"   3. Develop conversion campaigns for high-income, low-spending segments")
print(f"   4. Use cluster characteristics for personalized marketing")

print(f"\n Exercise completed successfully!")
print(f"   All required tasks have been accomplished:")
print(f"    Applied clustering on Annual Income and Spending Score")
print(f"    Visualized clusters in 2D plot with different colors")
print(f"    Identified high-potential customers")
print(f"    Calculated average distances from centroids")
