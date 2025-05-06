import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar datos procesados
df = pd.read_csv("../data/processed/dataset_clean.csv")

# Definir características y variable objetivo
features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 
            'loudness', 'mode', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 
            'time_signature', 'track_genre_encoded']

X = df[features]
y = df['popularity']

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Lista de modelos a probar
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "SVR": SVR()
}

# Carpeta para guardar modelos
models_dir = "../models/"
os.makedirs(models_dir, exist_ok=True)

# Evaluar cada modelo y guardar sus resultados
mejor_modelo = None
mejor_rmse = float("inf")

print("\nEntrenando modelos...")
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(nombre, "- RMSE:", round(rmse, 1), "- MAE:", round(mae, 1), "- R²:", round(r2, 3))

    # Guardar modelo en la carpeta "models"
    modelo_path = os.path.join(models_dir, f"{nombre.replace(' ', '_')}.pkl")
    joblib.dump(modelo, modelo_path)

    # Actualizar el mejor modelo según RMSE
    if rmse < mejor_rmse:
        mejor_rmse = rmse
        mejor_modelo = modelo_path

print("\nMejor modelo seleccionado:", mejor_modelo)

# Optimización del mejor modelo
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Cargar el mejor modelo encontrado
modelo_final = joblib.load(mejor_modelo)

search = RandomizedSearchCV(modelo_final, param_dist, n_iter=3, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=2)
search.fit(X_train, y_train)

print("\nMejor modelo optimizado:", search.best_params_)
print("Error optimizado (RMSE):", round(np.sqrt(-search.best_score_), 1))

# Guardar el modelo final
modelo_final_path = os.path.join(models_dir, "modelo_final.pkl")
joblib.dump(search.best_estimator_, modelo_final_path)
print("Modelo final guardado en:", modelo_final_path)
