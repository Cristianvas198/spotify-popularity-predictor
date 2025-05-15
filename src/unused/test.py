# train.py

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
from sklearn.cluster import KMeans

# Configuración
SEED = 42
os.makedirs("../data/train/", exist_ok=True)
os.makedirs("../data/test/", exist_ok=True)
os.makedirs("../models/", exist_ok=True)

def load_and_prepare_data():
    df = pd.read_csv("../data/processed/dataset_clean.csv")

    features = ['explicit', 'danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'duration_min', 'track_genre_encoded']

    X = df[features]
    y = df["popularity"]

    return X, y, features

def create_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(random_state=SEED),
        "SVR": SVR()
    }

def train_supervised_models(X_train, X_test, y_train, y_test):
    modelos = get_models()
    best_model_name = None
    best_rmse = float("inf")
    best_pipeline = None

    for name, model in modelos.items():
        print(f"\nEntrenando modelo: {name}")
        pipe = create_pipeline(model)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"{name} - RMSE: {rmse:.2f}, R2: {r2:.2f}")

        joblib.dump(pipe, f"../models/{name}.pkl")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_pipeline = pipe

    return best_model_name, best_pipeline, best_rmse

def optimize_best_model(X_train, y_train, X_test, y_test, base_model_name):
    print("Realizando búsqueda aleatoria de hiperparámetros...")

    param_dist = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    model_class = get_models()[base_model_name]
    pipeline = create_pipeline(model_class)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=SEED
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    joblib.dump(best_model, "../models/best_model.pkl")

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mejores parámetros encontrados: {search.best_params_}")
    print(f"Modelo optimizado - RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print("Modelo optimizado guardado como best_model.pkl.")

def train_unsupervised_model(X):
    print("\nEntrenando modelo no supervisado (KMeans)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=SEED)
    clusters = kmeans.fit_predict(X_scaled)

    joblib.dump(kmeans, "../models/kmeans.pkl")
    print("Modelo KMeans guardado como kmeans.pkl.")
    
    df_clusters = pd.DataFrame(X)
    df_clusters["cluster"] = clusters
    df_clusters.to_csv("../data/train/clusters.csv", index=False)
    print("Cluster assignments guardados en clusters.csv.")

def main():
    X, y, features = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    best_model_name, _, best_rmse = train_supervised_models(X_train, X_test, y_train, y_test)
    print(f"\nMejor modelo: {best_model_name} con RMSE: {best_rmse:.2f}")

    if best_model_name in ["RandomForest", "GradientBoosting", "XGBoost"]:
        optimize_best_model(X_train, y_train, X_test, y_test, best_model_name)

    train_unsupervised_model(X)

if __name__ == "__main__":
    main()

