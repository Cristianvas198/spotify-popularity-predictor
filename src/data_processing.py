import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Limpieza inicial
print("Cargando y limpiando datos...")
df = pd.read_csv("../data/raw/dataset.csv", index_col=0)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 2. Preprocesamiento básico
df["explicit"] = df["explicit"].astype(int)
df["duration_min"] = df["duration_ms"] / 60000
le = LabelEncoder()
df["track_genre_encoded"] = le.fit_transform(df["track_genre"])

# 3. Ingeniería de características (simplificada)
df['energy_loudness'] = df['energy'] * df['loudness']
df['dance_valence'] = df['danceability'] * df['valence']
df['speech_to_acoustic'] = df['speechiness'] / (df['acousticness'] + 0.001)

# 4. Eliminar columnas no necesarias y guardar
df.drop(columns=["track_id", "artists", "album_name", "track_name", "track_genre", "duration_ms"], inplace=True)
df.to_csv("../data/processed/dataset_clean.csv", index=False)
print("Datos preprocesados guardados.")