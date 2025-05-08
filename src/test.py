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
features=['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
 'duration_min', 'track_genre_encoded']

# features = ['duration_min', 'explicit', 'danceability', 'energy', 'key', 
#              'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness', 'liveness', 'valence', 'tempo','track_genre_encoded']

X = df.drop(columns=["popularity"])
y = df['popularity']

# Normalización de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Restaurar nombres de columna después de la normalización
X_scaled_df = df.drop(columns=["popularity", "time_signature"])
# X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Verificar nombres de columna antes de entrenar el modelo
print("Columnas en X_train:", X_train.columns)

# Crear las carpetas si no existen
train_dir = "../data/train/"
test_dir = "../data/test/"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Guardar los conjuntos de datos
df_train = pd.DataFrame(X_train, columns=features)
df_train["popularity"] = y_train
df_train.to_csv(os.path.join(train_dir, "train_data.csv"), index=False)

df_test = pd.DataFrame(X_test, columns=features)
df_test["popularity"] = y_test
df_test.to_csv(os.path.join(test_dir, "test_data.csv"), index=False)

print("\nDatos de entrenamiento guardados en:", os.path.join(train_dir, "train_data.csv"))
print("Datos de prueba guardados en:", os.path.join(test_dir, "test_data.csv"))

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
mejor_modelo_nombre = None
mejor_modelo_instancia = None
mejor_rmse = float("inf")

print("\nEntrenando modelos...")
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mae = mean_absolute_error(y_test, predicciones)
    r2 = r2_score(y_test, predicciones)

    print(f"{nombre} - RMSE: {round(rmse, 1)} - MAE: {round(mae, 1)} - R²: {round(r2, 3)}")

    # Guardar modelo en la carpeta "models"
    modelo_path = os.path.join(models_dir, f"{nombre.replace(' ', '_')}.pkl")
    joblib.dump(modelo, modelo_path)

    # Actualizar el mejor modelo según RMSE
    if rmse < mejor_rmse:
        mejor_rmse = rmse
        mejor_modelo_nombre = nombre
        mejor_modelo_instancia = modelo

print("\nMejor modelo seleccionado:", mejor_modelo_nombre)

# Verificar nombres de características en el mejor modelo antes de guardarlo
print("Características usadas en el modelo:", mejor_modelo_instancia.feature_names_in_)

# Optimización del mejor modelo (solo si admite hiperparámetros)
if isinstance(mejor_modelo_instancia, (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor)):
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    search = RandomizedSearchCV(mejor_modelo_instancia, param_dist, n_iter=3, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=2)
    search.fit(X_train, y_train)

    print("\nMejor modelo optimizado:", search.best_params_)
    print("Error optimizado (RMSE):", round(np.sqrt(-search.best_score_), 1))

    # Guardar el modelo final
    modelo_final_path = os.path.join(models_dir, "modelo_final.pkl")
    joblib.dump(search.best_estimator_, modelo_final_path)
    print("Modelo final guardado en:", modelo_final_path)
else:
    print("\nEl modelo seleccionado no admite optimización con hiperparámetros.")