import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Limpieza inicial
print("Cargando y limpiando datos...")
df = pd.read_csv("../data/raw/dataset.csv", index_col=0)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

#Preprocesamiento básico
df["explicit"] = df["explicit"].astype(int)
df["duration_min"] = df["duration_ms"] / 60000

#Aplicar Label Encoding a la columna 'track_genre'
print("Codificando la columna 'track_genre'...")
le = LabelEncoder()
df["track_genre_encoded"] = le.fit_transform(df["track_genre"])

#Guardar el mapeo de géneros
genre_mapping = pd.DataFrame({"Genre": le.classes_, "Encoded_Value": range(len(le.classes_))})
genre_mapping.to_csv("../data/processed/track_genre_mapping.csv", index=False)
print("Archivo 'track_genre_mapping.csv' guardado correctamente.")

#Data eng (simplificada)
df['energy_loudness'] = df['energy'] * df['loudness']
df['dance_valence'] = df['danceability'] * df['valence']
df['speech_to_acoustic'] = df['speechiness'] / (df['acousticness'] + 0.001)

#Eliminar columnas no necesarias y guardar
df.drop(columns=["track_id", "artists", "album_name", "track_name", "track_genre", "duration_ms"], inplace=True)
df.to_csv("../data/processed/dataset_clean.csv", index=False)
print("Datos preprocesados guardados.")
