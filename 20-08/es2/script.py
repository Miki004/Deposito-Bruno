import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("20-08\\es2\\dataset\\rsc\\Online_Retail_utf.csv")
print(df.columns.to_list())

df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
clv = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
clv.rename(columns={'TotalPrice':'CustomerLifetimeValue'}, inplace=True)
df = df.merge(clv, on='CustomerID', how='left')

df.groupby("CustomerID").agg({
    'Quantity' : "sum",
    'CustomerLifetimeValue': "mean",
    'InvoiceDate': "count",
    'Country': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'

})

standard = StandardScaler()

feature_target = df[["Quantity", "CustomerLifetimeValue"]]
X_scaled = standard.fit_transform(feature_target)

wcss = [] 
for k in range(1, 15): # proviamo diversi numeri di cluster
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42) # init='k-means++': inizializzazione “intelligente” dei centroidi, random_state=42: seme fisso per avere sempre gli stessi risultati (riproducibilità)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
 
plt.plot(range(1, 15), wcss, marker='o') # plotto sull'asse x i vari numeri di cluster, sull'asse y il valore si wcss ottenuto per ogni cluster
plt.xlabel('Numero di cluster')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the dataframe
df['Cluster'] = cluster_labels

# Plot the clusters
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    X_scaled[:, 0],  
    X_scaled[:, 1], 
    c=cluster_labels,
    cmap='tab10', 
    s=30,
    alpha=0.7
)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    c='black', s=200, marker='X', label='Centroids'
)

plt.xlabel('Quantity (scaled)')
plt.ylabel('CustomerLifetimeValue (scaled)')
plt.title('Customer Clusters (k = {})'.format(optimal_k))
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.show()


