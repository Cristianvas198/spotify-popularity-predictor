import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Cargar modelo y encoder
modelo = joblib.load('modelo_popularidad.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Título de la app
st.title("🎵 Predicción de Popularidad de una Canción")

st.markdown("Introduce las características de tu canción para predecir su nivel de popularidad.")

# Formulario de entrada
duration_ms = st.slider('Duración (ms)', 30000, 600000, 200000)
explicit = st.selectbox('¿Es explícita?', ['Sí', 'No'])
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
key = st.selectbox('Tonalidad (Key)', list(range(12)))
loudness = st.slider('Loudness (dB)', -60.0, 0.0, -10.0)
mode = st.selectbox('Modo', ['Menor (0)', 'Mayor (1)'])
speechiness = st.slider('Speechiness', 0.0, 1.0, 0.05)
acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.0)
liveness = st.slider('Liveness', 0.0, 1.0, 0.2)
valence = st.slider('Valence', 0.0, 1.0, 0.5)
tempo = st.slider('Tempo (BPM)', 40.0, 250.0, 120.0)
time_signature = st.selectbox('Compás (Time Signature)', [1, 2, 3, 4, 5, 6, 7])

# Input del género
track_genre = st.selectbox('Género de la canción', label_encoder.classes_)
track_genre_encoded = label_encoder.transform([track_genre])[0]

# Botón de predicción
if st.button('🎧 Predecir Popularidad'):
    entrada = pd.DataFrame([[
        duration_ms,
        1 if explicit == 'Sí' else 0,
        danceability,
        energy,
        key,
        loudness,
        1 if mode == 'Mayor (1)' else 0,
        speechiness,
        acousticness,
        instrumentalness,
        liveness,
        valence,
        tempo,
        time_signature,
        track_genre_encoded
    ]], columns=[
        'duration_ms', 'explicit', 'danceability', 'energy', 'key',
        'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature', 'track_genre_encoded'
    ])

    prediccion = modelo.predict(entrada)[0]
    prediccion_redondeada = int(round(prediccion))

    st.subheader(f'🎯 Popularidad Estimada: {prediccion_redondeada}/100')

    # Interpretación simple
    if prediccion_redondeada >= 80:
        st.success("🔥 ¡Potencial hit! Esta canción podría ser muy popular.")
    elif prediccion_redondeada >= 60:
        st.info("👍 Buena canción, tiene potencial para sonar bastante.")
    elif prediccion_redondeada >= 40:
        st.warning("🧐 Puede que necesite promoción para destacar.")
    else:
        st.error("🙁 Baja popularidad estimada. ¿Quizás otro enfoque?")
