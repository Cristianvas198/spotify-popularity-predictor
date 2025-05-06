import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Cargar datos sin procesar
print("Cargando datos...")
df = pd.read_csv("../data/raw/dataset.csv", index_col=0)

print("Total de filas:", df.shape[0])
print("Total de columnas:", df.shape[1])

# Eliminar valores nulos y duplicados
print("\nEliminando valores nulos y duplicados...")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convertir columna 'explicit' de booleano a entero
df['explicit'] = df['explicit'].astype(int)

# Codificar 'track_genre' en valores numéricos
le = LabelEncoder()
df['track_genre_encoded'] = le.fit_transform(df['track_genre'])

# Guardar datos limpios en data/processed/
processed_path = "../data/processed/dataset_clean.csv"
df.to_csv(processed_path, index=False)
print("\nDatos procesados guardados en:", processed_path)
