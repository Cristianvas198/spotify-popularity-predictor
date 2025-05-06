import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar el modelo final entrenado
modelo_final_path = "../models/modelo_final.pkl"
modelo = joblib.load(modelo_final_path)

print("\nModelo cargado correctamente:", modelo_final_path)

# Cargar los datos de prueba
df_test = pd.read_csv("../data/processed/dataset_clean.csv")

features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 
            'loudness', 'mode', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 
            'time_signature', 'track_genre_encoded']

X_test = df_test[features]
y_test = df_test['popularity']

# Predicción
y_pred = modelo.predict(X_test)

# Evaluación del modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nResultados de la evaluación:")
print("RMSE:", round(rmse, 1))
print("MAE:", round(mae, 1))
print("R²:", round(r2, 3))
