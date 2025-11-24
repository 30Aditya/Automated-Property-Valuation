import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# ------ SAMPLE DATA ------
data = {
    'area_sqft': [1000, 1500, 2000, 1800, 2400, 3000, 3500, 4000],
    'bedrooms': [2, 3, 3, 3, 4, 4, 4, 5],
    'age_years': [10, 15, 20, 18, 8, 5, 3, 2],
    'price_lakhs': [40, 55, 70, 65, 90, 110, 130, 150]
}

df = pd.DataFrame(data)

# Features for clustering
X = df[['area_sqft', 'bedrooms', 'age_years']]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------ ELBOW METHOD GRAPH ------
inertia = []
K = range(1, 7)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid()
plt.show()

# ------ K-MEANS MODEL ------
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# ------ Find average price of each cluster ------
cluster_prices = df.groupby('Cluster')['price_lakhs'].mean()
print("Cluster-wise Average Prices:\n", cluster_prices)

# ------ SCATTER PLOT: AREA vs PRICE ------
plt.scatter(df['area_sqft'], df['price_lakhs'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price Clusters (K-Means)")
plt.colorbar(label='Cluster')
plt.show()

# ------ 3D CLUSTER PLOT ------
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['area_sqft'], df['bedrooms'], df['age_years'],
           c=df['Cluster'], cmap='viridis', s=80)

ax.set_xlabel("Area (sqft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Age (years)")
ax.set_title("3D Visualization of K-Means Clusters")

plt.show()

# ------ Predicting for NEW HOUSE ------
new_house = [[2600, 4, 6]]  # area, bedrooms, age
new_house_scaled = scaler.transform(new_house)

cluster = kmeans.predict(new_house_scaled)[0]
estimated_price = cluster_prices[cluster]

print("\nThe new house belongs to Cluster:", cluster)
print("Estimated Price Range:", estimated_price, "Lakhs")