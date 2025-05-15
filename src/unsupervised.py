import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Cargando datos limpios...")
df = pd.read_csv("../data/processed/dataset_clean.csv")

features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_min', 'track_genre_encoded']

X = df[features]

print("Escalando datos...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Aplicando K-Means con 5 clusters...")
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["cluster"] = clusters
df.to_csv("../data/processed/dataset_clustered.csv", index=False)
joblib.dump(kmeans, "../models/kmeans_model.pkl")

print("Reducción de dimensiones con PCA para visualización...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["cluster"] = clusters

plt.figure(figsize=(8,6))
for c in df_plot["cluster"].unique():
    subset = df_plot[df_plot["cluster"] == c]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {c}")
plt.title("Clusters de canciones (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("../data/processed/clusters_visualization.png")
plt.close()

print("Clusterización completada y guardada.")
