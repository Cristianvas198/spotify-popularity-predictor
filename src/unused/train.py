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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


SEED = 42 
os.makedirs("../data/train/", exist_ok=True)
os.makedirs("../data/test/", exist_ok=True)
os.makedirs("../models/", exist_ok=True)


def load_and_prepare_data(): #Cargar y preparar los datos
    df = pd.read_csv("../data/processed/dataset_clean.csv")

    features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_min', 'track_genre_encoded']
    
    X = df[features]
    y = df["popularity"]

    return X, y, features

def create_preprocessor(): #Pipeline para preprocesar los datos
    return Pipeline([("scaler", StandardScaler())])


def get_models(): #Modelos a probar
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(random_state=SEED),
        "SVR": SVR()
    }

def train_and_evaluate(): #Entrenamiento y evaluación de modelos
    X, y, features = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    
    preprocessor = create_preprocessor() # Preprocesamiento
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "../models/preprocessor.pkl") # Guardar scaler


    X_train_df = pd.DataFrame(X_train_preprocessed, columns=features) # Convertir a DataFrame manteniendo los nombres de columnas
    X_test_df = pd.DataFrame(X_test_preprocessed, columns=features)


    X_train_df.to_csv("../data/train/X_train.csv", index=False) # Guardar datasets
    pd.DataFrame(y_train).to_csv("../data/train/y_train.csv", index=False)
    X_test_df.to_csv("../data/test/X_test.csv", index=False)
    pd.DataFrame(y_test).to_csv("../data/test/y_test.csv", index=False)

    modelos = get_models() # Entrenar modelos
    best_model = None
    best_score = float("inf")

    for name, model in modelos.items():
        print(f"\nEntrenando {name}...")
        model.fit(X_train_df, y_train)

        y_pred = model.predict(X_test_df) # Evaluación
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")

        joblib.dump(model, f"../models/{name}.pkl") # Guardar modelo

        if rmse < best_score: # Actualizar mejor modelo
            best_score = rmse
            best_model = name

    print(f"\nMejor modelo: {best_model} con RMSE: {best_score:.2f}")

    if best_model in ["RandomForest", "GradientBoosting", "XGBoost"]: # Optimización del mejor modelo
        print("\nOptimizando el mejor modelo...")
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        best_model_instance = modelos[best_model]
        search = RandomizedSearchCV(
            best_model_instance,
            param_dist,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=SEED,
        )

        search.fit(X_train_df, y_train)
        print(f"Mejores parámetros: {search.best_params_}")

        joblib.dump(search.best_estimator_, "../models/best_model.pkl") # Guardar modelo optimizado
        print("Modelo optimizado guardado.")

if __name__ == "__main__":
    train_and_evaluate()
