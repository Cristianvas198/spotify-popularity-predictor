import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

modelo_final_path = "../models/best_model.pkl" # Cargar el modelo final entrenado
modelo = joblib.load(modelo_final_path)

print("\nModelo cargado correctamente:", modelo_final_path)

df_test = pd.read_csv("../data/processed/dataset_clean.csv") # Cargar los datos de prueba

features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness',
            'valence', 'tempo', 'duration_min', 'track_genre_encoded']

X_test = df_test[features]
y_test = df_test["popularity"]

print("Columnas en X_test:", X_test.columns.tolist()) # Verificar que los nombres de las columnas coincidan antes de hacer la predicción
print("Columnas esperadas por el modelo:", modelo.feature_names_in_)

y_pred = modelo.predict(X_test) # Predicción

rmse = np.sqrt(mean_squared_error(y_test, y_pred)) # Evaluación del modelo
r2 = r2_score(y_test, y_pred)

print("\nResultados de la evaluación:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

