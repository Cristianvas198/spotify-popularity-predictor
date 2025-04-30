# spotify-popularity-predictor

## Descripción
Este proyecto aplica **Machine Learning** para predecir la **valoración** de una canción en **Spotify** basándose en sus características, como el ritmo, la energía, la acústica y otros atributos clave.

## Objetivo
El propósito es desarrollar un modelo que prediga la popularidad de una canción según sus propiedades extraídas de la **Spotify API**. Esto puede ayudar a **artistas, productores y analistas** a comprender mejor qué factores influyen en el éxito de una canción.

## Estructura del Proyecto

```
|-- nombre_proyecto_final_ML
    |-- data
    |   |-- raw
    |        |-- dataset.csv
    |        |-- ...
    |   |-- processed
    |   |-- train
    |   |-- test
    |
    |-- notebooks
    |   |-- 01_Fuentes.ipynb
    |   |-- 02_LimpiezaEDA.ipynb
    |   |-- 03_Entrenamiento_Evaluacion.ipynb
    |   |-- ...
    |
    |-- src
    |   |-- data_processing.py
    |   |-- training.py
    |   |-- evaluation.py
    |   |-- ...
    |
    |-- models
    |   |-- trained_model.pkl
    |   |-- model_config.yaml
    |   |-- ...
    |
    |-- app_streamlit
    |   |-- app.py
    |   |-- requirements.txt
    |   |-- ...
    |
    |-- docs
    |   |-- negocio.ppt
    |   |-- ds.ppt
    |   |-- memoria.md
    |   |-- ...
    |
    |
    |-- README.md

```