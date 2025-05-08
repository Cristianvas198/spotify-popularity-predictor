import streamlit as st
import pandas as pd
import joblib
import time

# Cargar el modelo
modelo_path = "../models/best_model.pkl"
modelo = joblib.load(modelo_path)
genre_df = pd.read_csv("../data/processed/track_genre_mapping.csv")
genre_dict = dict(zip(genre_df["track_genre"], genre_df["track_genre_encoded"]))

# =============================================
# ESTILOS COMPLETOS
# =============================================
st.markdown("""
<style>
/* Fondo principal */
.stApp {
    background-color: #12122A;
    font-family: 'Arial', sans-serif;
}

/* Texto general blanco */
* {
    color: white !important;
}

/* Contenedores principales */
.stNumberInput, .stTextInput, .stSelectbox, .stRadio {
    border: none !important;
    box-shadow: none !important;
}

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox select {
    background-color: #252545 !important;
    border: 1px solid #4a4a8a !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
}

/* Sliders - valor actual (ROJO) */
.stSlider [data-testid="stThumbValue"] {
    color: red !important;
    background-color: transparent !important;
    font-weight: bold;
    padding: 0;
    margin-top: -15px;
}

/* Sliders - track */
.stSlider [role="slider"] {
    background: linear-gradient(90deg, #6e00ff 0%, #8a2be2 100%) !important;
    border: none !important;
}

/* Radio buttons */
.stRadio > div {
    justify-content: center;
    gap: 20px;
    background-color: transparent !important;
    border: none !important;
}

/* Header */
.header {
    background: linear-gradient(135deg, #6e00ff 0%, #12122A 100%);
    padding: 25px;
    border-radius: 0 0 15px 15px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.logo {
    font-size: 64px;
    font-weight: 900;
    margin: 5px 0;
    text-shadow: 0 0 10px rgba(110, 0, 255, 0.7);
}

/* Secciones */
.input-section {
    background-color: #1E1E3A;
    padding: 20px;
    border-radius: 12px;
    margin: 15px auto;
    max-width: 750px;
    border-left: 3px solid #6e00ff;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.section-title {
    font-size: 20px;
    text-align: center;
    margin-bottom: 15px;
    padding-bottom: 5px;
    border-bottom: 1px solid #6e00ff;
}

/* Botón */
div.stButton > button:first-child {
    display: block;
    margin: 20px auto;
    width: 240px;
    background: linear-gradient(135deg, #6e00ff 0%, #8a2be2 100%);
    font-weight: bold;
    padding: 14px;
    border-radius: 8px;
    border: none;
    transition: all 0.3s;
    font-size: 16px;
    box-shadow: 0 4px 8px rgba(110, 0, 255, 0.3);
}

/* Resultados */
.result-box {
    background-color: #1E1E3A;
    padding: 20px;
    border-radius: 12px;
    margin: 20px auto;
    max-width: 750px;
    text-align: center;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    border-top: 3px solid #6e00ff;
}

/* Predicción en ROJO */
.prediccion-valor {
    color: #FF5555 !important;
    font-size: 36px;
    margin: 10px 0;
}

.prediccion-categoria {
    color: #FF5555 !important;
    font-size: 24px;
    margin-bottom: 15px;
}

.recommendation-box {
    background-color: #252545;
    padding: 15px;
    border-radius: 8px;
    margin: 15px auto;
    max-width: 750px;
    border-left: 3px solid #6e00ff;
}

.recommendation-title {
    font-weight: bold;
    font-size: 18px;
    margin-bottom: 10px;
    text-align: center;
    color: #6e00ff !important;
}

.recommendation-item {
    margin: 8px 0;
    padding-left: 15px;
    position: relative;
}

.recommendation-item:before {
    content: "•";
    color: #6e00ff;
    font-weight: bold;
    position: absolute;
    left: 0;
}
</style>
""", unsafe_allow_html=True)

# =============================================
# INTERFAZ DE LA APLICACIÓN
# =============================================

# Encabezado
st.markdown("""
<div class="header">
    <div class="logo">NØIZE</div>
    <div class="tagline">No todo lo que suena es NØIZE... pero lo que sí, lo sabrás aquí</div>
</div>
""", unsafe_allow_html=True)

# Sección de información básica
with st.container():
    st.markdown("""
    <div class="input-section">
    <div class="section-title">INFORMACIÓN BÁSICA</div>
    """, unsafe_allow_html=True)
    
    nombre_artista = st.text_input("Artista", placeholder="Ej: Bad Bunny", key="artist")
    nombre_cancion = st.text_input("Canción", placeholder="Ej: Dakiti", key="song")

    if not nombre_cancion:
        nombre_cancion = "Canción desconocida"
    if not nombre_artista:
        nombre_artista = "Artista desconocido"
    
    st.markdown("</div>", unsafe_allow_html=True)

# Sección de características musicales
with st.container():
    st.markdown("""
    <div class="input-section">
    <div class="section-title">CARACTERÍSTICAS MUSICALES</div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Duración**")
        duracion_cols = st.columns(2)
        with duracion_cols[0]:
            minutos = st.number_input("Minutos", min_value=0, max_value=10, value=3, key="mins")
        with duracion_cols[1]:
            segundos = st.number_input("Segundos", min_value=0, max_value=59, value=30, key="secs")
        duracion_total = minutos + (segundos/60)
        
        explicito = st.radio("Contenido explícito", ["Sí", "No"], horizontal=True)
        tonalidad = st.selectbox("Tonalidad", list(range(0, 12)))
        tempo = st.number_input("Tempo (BPM)", min_value=60, max_value=200, value=120)

    with col2:
        nivel_bailabilidad = st.selectbox("Bailabilidad", ["Bajo", "Medio", "Alto"])
        nivel_energia = st.selectbox("Energía", ["Bajo", "Medio", "Alto"])
        volumen = st.number_input("Volumen", min_value=0.0, max_value=60.0, value=5.0, step=0.5)
        valor_volumen = -abs(volumen)
        modo = st.radio("Modo", ["Mayor", "Menor"], horizontal=True)
        genero_musical = st.selectbox("Género", list(genre_dict.keys()))
    
    # Sliders
    acustica = st.slider("Acústica", min_value=0.0, max_value=1.0, value=0.3)
    instrumentalidad = st.slider("Instrumentalidad", min_value=0.0, max_value=1.0, value=0.0)
    vivacidad = st.slider("Vivacidad", min_value=0.0, max_value=1.0, value=0.5)
    verbalizacion = st.slider("Verbalización", min_value=0.0, max_value=1.0, value=0.1)
    valencia = st.slider("Valencia", min_value=0.0, max_value=1.0, value=0.5)

    st.markdown("</div>", unsafe_allow_html=True)

# Procesamiento de datos
genero_codificado = genre_dict[genero_musical]
modo = 1 if modo == "Mayor" else 0
explicito = 1 if explicito == "Sí" else 0
bailabilidad = {"Bajo": 0.2, "Medio": 0.5, "Alto": 0.8}[nivel_bailabilidad]
energia = {"Bajo": 0.2, "Medio": 0.5, "Alto": 0.8}[nivel_energia]

# Construcción del DataFrame
orden_correcto = modelo.feature_names_in_
data = pd.DataFrame([[explicito, bailabilidad, energia, tonalidad, valor_volumen, modo, 
                     acustica, instrumentalidad, vivacidad, verbalizacion, 
                     valencia, genero_codificado, duracion_total, tempo]], 
                   columns=orden_correcto)

# Validación
if any(pd.isnull(data.iloc[0])):
    st.error("❌ Por favor complete todos los campos")
    st.stop()

# Botón de predicción
button = st.button("PREDECIR POPULARIDAD")

if button:
    with st.spinner("🔍 Analizando la esencia musical de tu canción..."):
        time.sleep(2)
        
    try:
        prediccion = modelo.predict(data)[0]
        
        # Resultado y recomendaciones
        if prediccion < 40:
            color = "#FF5555"
            nivel = "BAJA POPULARIDAD"
            mensaje = "Tu canción necesita ajustes para destacar en el mercado actual"
            recomendaciones = [
                "Aumenta la energía y bailabilidad para hacerla más atractiva",
                "Experimenta con diferentes géneros musicales",
                "Invierte en marketing digital y redes sociales",
                "Colabora con otros artistas para aumentar tu alcance",
                "Considera trabajar con un productor experimentado"
            ]
        elif 40 <= prediccion < 70:
            color = "#FFAA44"
            nivel = "POPULARIDAD MODERADA"
            mensaje = "Tienes una base sólida con potencial para crecer"
            recomendaciones = [
                "Optimiza tu estrategia de lanzamiento con un plan de marketing",
                "Considera lanzar un remix o versión acústica",
                "Enfócate en plataformas como Spotify y TikTok",
                "Analiza canciones similares con mejor desempeño",
                "Mejora la calidad de producción"
            ]
        else:
            color = "#6e00ff"
            nivel = "ALTA POPULARIDAD"
            mensaje = "¡Tienes un hit potencial entre manos!"
            recomendaciones = [
                "Planifica una campaña de lanzamiento profesional",
                "Considera sincronizaciones en medios y publicidad",
                "Prepara material visual de alta calidad",
                "Negocia con sellos discográficos o distribuidores",
                "Organiza una gira o presentaciones en vivo"
            ]
            
        # Mostrar resultados
        st.markdown(f"""
        <div class="result-box">
            <h2>{nombre_cancion if nombre_cancion else 'Canción desconocida'} - {nombre_artista if nombre_artista else 'Artista desconocido'}</h2>
            <div class="prediccion-valor">{prediccion:.0f}/100</div>
            <div class="prediccion-categoria">{nivel}</div>
            <p>{mensaje}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar recomendaciones
        st.markdown(f"""
        <div class="recommendation-box">
            <div class="recommendation-title">RECOMENDACIONES PARA TU CANCIÓN</div>
        """, unsafe_allow_html=True)
        
        for recomendacion in recomendaciones:
            st.markdown(f"""
            <div class="recommendation-item">{recomendacion}</div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error en el análisis: {str(e)}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; color: #6e00ff; font-size: 11px;">
    NØIZE Predictor v1.0 | Herramienta profesional para artistas y productores
</div>
""", unsafe_allow_html=True)