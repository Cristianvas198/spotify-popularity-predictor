import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
print("Cargando datos...")
df = pd.read_csv("../data/processed/dataset_clean.csv")
features = [col for col in df.columns if col != "popularity"]
X = df[features]
y = df["popularity"]

# 2. Dividir datos (para comparación directa con K-Fold)
print("Dividiendo datos en entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Definir el modelo
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1  # Usar todos los núcleos del CPU
)

# 4. Aplicar K-Fold Cross Validation
print("\nEvaluando con K-Fold Cross Validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_squared_error"))

# Mostrar resultados de K-Fold
print("\nResultados K-Fold:")
print(f"RMSE promedio: {rmse_scores.mean():.2f}")
print(f"RMSE desviación estándar: {rmse_scores.std():.2f}")

# 5. Entrenar el modelo final con todo el set de entrenamiento
print("\nEntrenando modelo final en todo el conjunto de entrenamiento...")
model.fit(X_train, y_train)

# 6. Evaluar en conjunto de prueba
y_pred = model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE manualmente
r2_test = r2_score(y_test, y_pred)

print("\nResultados en conjunto de prueba:")
print(f"RMSE: {rmse_test:.2f}")
print(f"R2: {r2_test:.2f}")

# 7. Guardar modelo entrenado
print("\nGuardando modelo...")
joblib.dump(model, "../models/model.pkl")
print("Modelo guardado en ../models/model.pkl")
