import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#Cargar datos
print("Cargando datos...")
df = pd.read_csv("../data/processed/dataset_clean.csv")
features = [col for col in df.columns if col != "popularity"]
X = df[features]
y = df["popularity"]

#Dividir datos
print("Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Entrenar modelo
print("Entrenando modelo...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Usar todos los núcleos del CPU
)
model.fit(X_train, y_train)

#Evaluar
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred)) #Calculamos RMSE manualmente
r2 = r2_score(y_test, y_pred)

print("\nResultados:")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")

#Guardar modelo
print("\nGuardando modelo...")
joblib.dump(model, "../models/model.pkl")
print("Modelo guardado en ../models/model.pkl")