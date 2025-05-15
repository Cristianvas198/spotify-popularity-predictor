import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_model_and_data():
    """Carga el modelo y los datos con manejo de errores"""
    try:
        model = joblib.load("../models/model.pkl")
        df = pd.read_csv("../data/processed/dataset_clean.csv")
        return model, df
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Asegúrate de haber ejecutado primero:")
        print("1. data_preprocessing.py")
        print("2. model_training.py")
        exit(1)

def prepare_features(df):
    """Prepara las características eliminando columnas no necesarias"""
    return [col for col in df.columns if col != "popularity"]

def evaluate_model(model, X, y):
    """Evalúa el modelo y muestra métricas"""
    y_pred = model.predict(X)
    
    # Cálculo compatible con todas las versiones de scikit-learn
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return rmse, r2

def main():
    print("Cargando modelo y datos...")
    model, df = load_model_and_data()
    
    print("Preparando características...")
    features = prepare_features(df)
    X = df[features]
    y = df["popularity"]
    
    print("Evaluando modelo...")
    rmse, r2 = evaluate_model(model, X, y)
    
    print("\nRESULTADOS FINALES:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.2f}")
    print("\nInterpretación:")
    print("- RMSE: Error cuadrático medio (menor es mejor)")
    print(f"- R2: El modelo explica el {r2*100:.1f}% de la variabilidad en la popularidad")

if __name__ == "__main__":
    main()