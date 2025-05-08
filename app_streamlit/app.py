import streamlit as st
import pandas as pd
import joblib
import time

# Cargar el modelo
modelo_path = "../models/best_model.pkl" # Carga el modelo final entrenado
modelo = joblib.load(modelo_path)

df = pd.read_csv("../data/processed/dataset_clean.csv") # Carga el dataset

genre_df = pd.read_csv("../data/processed/track_genre_mapping.csv") # Carga los generos codificados guardados en el procesamiento de datos
genre_dict = dict(zip(genre_df["track_genre"], genre_df["track_genre_encoded"]))

#**Diseño llamativo**
st.markdown("""
    <style>
    body {background-color: #121212; color: white; font-family: 'Arial', sans-serif;}
    .title {color: #1DB954; text-align: center; font-size: 42px; font-weight: bold;}
    .subtitle {color: #bbb; text-align: center; font-size: 24px;}
    .result-box {border-radius:10px; padding:20px; text-align:center; background-color: #222; color: white;}
    .result-value {font-size:36px; font-weight:bold;}
    .center-button {display: flex; justify-content: center; margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

#**Título**
st.markdown("<h1 class='title'>🎶 ¿Qué tan viral será tu canción? 🎤💥</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subtitle'>Descubre si tu tema puede conquistar el mundo 🌎✨</h2>", unsafe_allow_html=True)
st.divider()

# **Ingreso del nombre del artista y la canción**
artist_name = st.text_input("Nombre del artista", placeholder="Ej: Taylor Swift")
song_name = st.text_input("Nombre de la canción", placeholder="Ej: Shake It Off")

# **Asegurar que los nombres no estén vacíos**
if not song_name:
    song_name = "Canción desconocida"
if not artist_name:
    artist_name = "Artista desconocido"

# **Sección de entrada de datos**
col1, col2 = st.columns(2)
with col1:
    duration_min = st.number_input("Duración (min)", min_value=1.0, max_value=10.0, value=3.5)
    explicit = st.radio("¿Tiene contenido explícito?", ["Sí", "No"])
    key = st.selectbox("Tonalidad (Key)", list(range(0, 12)))
    tempo = st.number_input("Tempo (BPM)", min_value=60, max_value=200, value=120)

with col2:
    danceability_level = st.selectbox("Danceability", ["Bajo", "Medio", "Alto"])
    energy_level = st.selectbox("Energy", ["Bajo", "Medio", "Alto"])
    loudness = st.number_input("Loudness (dB)", min_value=-60.0, max_value=0.0, value=-5.0, step=0.5)
    mode = st.radio("Modo", ["Mayor", "Menor"])
    track_genre_name = st.selectbox("Género de la canción", list(genre_dict.keys()))

acousticness = st.slider("Acousticness", min_value=0.0, max_value=1.0, value=0.3)
instrumentalness = st.slider("Instrumentalness", min_value=0.0, max_value=1.0, value=0.0)
liveness = st.slider("Liveness", min_value=0.0, max_value=1.0, value=0.5)
speechiness = st.slider("Speechiness", min_value=0.0, max_value=1.0, value=0.1)
valence = st.slider("Valence", min_value=0.0, max_value=1.0, value=0.5)

track_genre_encoded = genre_dict[track_genre_name]

#**Convertir valores categóricos y niveles**
mode = 1 if mode == "Mayor" else 0
explicit = 1 if explicit == "Sí" else 0

# Conversión de niveles a valores numéricos
danceability = {"Bajo": 0.2, "Medio": 0.5, "Alto": 0.8}[danceability_level]
energy = {"Bajo": 0.2, "Medio": 0.5, "Alto": 0.8}[energy_level]

# **Construcción del DataFrame con el orden correcto**
orden_correcto = modelo.feature_names_in_
data = pd.DataFrame([[explicit, danceability, energy, key, loudness, mode,
                      acousticness, instrumentalness, liveness, speechiness, valence,
                      track_genre_encoded, duration_min, tempo]], 
                     columns=orden_correcto)

# Validación de inputs vacíos
if any(pd.isnull(data.iloc[0])):
    st.error("❌ Error: Todos los campos deben estar completos.")
    st.stop()

# **Botón grande y centrado con animación**
st.markdown('<div class="center-button">', unsafe_allow_html=True)
colA, colB, colC = st.columns([3, 3, 3])
with colB:
    button = st.button("🎵 Analizar potencial de éxito")
if button:
    with st.spinner("🎶 Calculando el potencial de éxito..."):
        time.sleep(2)  # Simulación de carga

    try:
        prediccion = modelo.predict(data)[0]

        # **Definir colores y mensajes según nivel de predicción**
        if prediccion < 40:
            color = "#FF4B4B"  # Rojo
            mensaje = "🚨 Popularidad baja: Mejora la promoción en redes y conéctate con tu audiencia."
        elif 40 <= prediccion < 70:
            color = "#FFA500"  # Naranja
            mensaje = "✨ Popularidad media: Colabora con otros artistas y usa playlists estratégicas."
        else:
            color = "#1DB954"  # Verde
            mensaje = "🏆 Popularidad alta: ¡Tu canción tiene potencial de éxito global!"

        # **Mostrar el resultado correctamente con el nombre del artista y canción**
        st.markdown(f"""
            <div class="result-box" style="background-color: {color};">
                <h2>🎶 {song_name} - {artist_name} 🎶</h2>
                <h3>🔥 Potencial de viralidad 🔥</h3>
                <h1 class="result-value">{prediccion:.2f}</h1>
                <p>{mensaje}</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error al hacer la predicción: {e}")
st.markdown('</div>', unsafe_allow_html=True)

