import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("Cargando datos...") # Cargar datos Raw
df = pd.read_csv("../data/raw/dataset.csv", index_col=0)

print("Total de filas:", df.shape[0])
print("Total de columnas:", df.shape[1])

print("\nEliminando valores nulos y duplicados...") # Eliminar valores nulos y duplicados
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df["explicit"] = df["explicit"].astype(int) #Conertir 'explicit' a int

df["duration_min"] = df["duration_ms"] / 60000 # Convertir duración de ms a minutos

le = LabelEncoder() # Codificar 'track_genre' en valores numéricos
df["track_genre_encoded"] = le.fit_transform(df["track_genre"])

genre_mapping = pd.DataFrame({"track_genre": df["track_genre"], "track_genre_encoded": df["track_genre_encoded"]})
genre_mapping.drop_duplicates().to_csv("../data/processed/track_genre_mapping.csv", index=False)

df.drop(columns=["track_id", "artists", "album_name", "track_name", "track_genre", "duration_ms"], inplace=True) # Eliminar columnas no necesarias

processed_path = "../data/processed/dataset_clean.csv" # Guardar datos limpios 
df.to_csv(processed_path, index=False)
print("\nDatos procesados guardados en:", processed_path)
print("Columnas finales:", df.columns.tolist())
