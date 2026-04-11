import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN DE LA INTERFAZ ---
st.set_page_config(page_title="App Estadística - Edgar Solis", layout="wide")

st.title("📊 Distribuciones y Prueba de Hipótesis")
st.markdown("""
Esta aplicación permite visualizar distribuciones de datos y realizar pruebas de hipótesis Z, 
integrando inteligencia artificial para la interpretación de resultados.
""")

# --- SIDEBAR: CONTROL DE DATOS ---
st.sidebar.header("Configuración de Datos")
n = st.sidebar.slider("Tamaño de muestra (n)", min_value=30, max_value=500, value=100)
st.sidebar.info("Nota: Se requiere n ≥ 30 para la Prueba Z.")

# Generación de datos sintéticos (Distribución Normal)
# Usamos session_state para que los datos no cambien cada vez que movamos un botón
if 'datos' not in st.session_state:
    st.session_state.datos = np.random.normal(loc=100, scale=15, size=n)

if st.sidebar.button("Generar nuevos datos"):
    st.session_state.datos = np.random.normal(loc=100, scale=15, size=n)

df = pd.DataFrame(st.session_state.datos, columns=['Valores'])

# --- MÓDULO DE VISUALIZACIÓN ---
st.header("1. Visualización de la Distribución")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma con KDE")
    fig, ax = plt.subplots()
    sns.histplot(df['Valores'], kde=True, ax=ax, color="skyblue")
    ax.set_title("Distribución de Frecuencias")
    st.pyplot(fig)

with col2:
    st.subheader("Boxplot (Caja y Bigotes)")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['Valores'], ax=ax, color="lightgreen")
    ax.set_title("Detección de Outliers y Cuartiles")
    st.pyplot(fig)

# Análisis automático básico
st.subheader("Análisis de la Muestra")
col_stat1, col_stat2, col_stat3 = st.columns(3)
col_stat1.metric("Media Muestral", f"{df['Valores'].mean():.2f}")
col_stat2.metric("Desviación Estándar", f"{df['Valores'].std():.2f}")
col_stat3.metric("Muestra (n)", len(df))