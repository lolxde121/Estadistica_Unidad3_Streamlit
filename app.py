import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="App Estadística - Edgar Solis", layout="wide")

st.title("📊 Distribuciones y Prueba de Hipótesis")
st.markdown("""
Esta aplicación permite visualizar distribuciones y realizar pruebas de hipótesis Z 
para una muestra con varianza poblacional conocida.
""")

# --- SIDEBAR: CONFIGURACIÓN DE DATOS ---
st.sidebar.header("1. Parámetros de la Muestra")
n = st.sidebar.slider("Tamaño de muestra (n)", 30, 500, 100)
mu_real = st.sidebar.number_input("Media real (para generar datos)", value=100.0)
sigma_pob = st.sidebar.number_input("Desviación estándar poblacional (σ)", value=15.0)

# Mantener los datos persistentes con session_state
if 'datos' not in st.session_state:
    st.session_state.datos = np.random.normal(loc=mu_real, scale=sigma_pob, size=n)

if st.sidebar.button("Regenerar Datos"):
    st.session_state.datos = np.random.normal(loc=mu_real, scale=sigma_pob, size=n)

df = pd.DataFrame(st.session_state.datos, columns=['Valores'])

# --- MÓDULO 1: VISUALIZACIÓN ---
st.header("1. Análisis Visual de la Distribución")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Histograma y KDE")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Valores'], kde=True, ax=ax1, color="skyblue")
    ax1.set_title("¿Es una distribución normal?")
    st.pyplot(fig1)

with col2:
    st.subheader("Boxplot")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=df['Valores'], ax=ax2, color="lightgreen")
    ax2.set_title("Identificación de Outliers")
    st.pyplot(fig2)

# --- MÓDULO 2: PRUEBA DE HIPÓTESIS Z ---
st.divider()
st.header("2. Prueba de Hipótesis Z")

col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Configuración de la Prueba")
    h0_val = st.number_input("Hipótesis Nula (H0: μ = )", value=100.0)
    alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

with col_p2:
    st.subheader("Tipo de Prueba")
    tipo_test = st.selectbox("Selecciona la Hipótesis Alternativa (H1)", 
                             ["Bilateral (μ ≠ H0)", "Cola Derecha (μ > H0)", "Cola Izquierda (μ < H0)"])

# --- CÁLCULOS MATEMÁTICOS ---
media_muestral = df['Valores'].mean()
# Fórmula Z = (media - H0) / (sigma / sqrt(n))
z_stat = (media_muestral - h0_val) / (sigma_pob / np.sqrt(n))

# Lógica para P-Value y Valores Críticos
if "Bilateral" in tipo_test:
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    z_critico = stats.norm.ppf(1 - alpha/2)
    rechazar = abs(z_stat) > z_critico
elif "Derecha" in tipo_test:
    p_value = 1 - stats.norm.cdf(z_stat)
    z_critico = stats.norm.ppf(1 - alpha)
    rechazar = z_stat > z_critico
else: # Izquierda
    p_value = stats.norm.cdf(z_stat)
    z_critico = stats.norm.ppf(alpha)
    rechazar = z_stat < z_critico

# --- RESULTADOS ---
st.subheader("Resultados del Análisis")
c1, c2, c3 = st.columns(3)
c1.metric("Estadístico Z", f"{z_stat:.4f}")
c2.metric("P-Value", f"{p_value:.4f}")

if rechazar:
    c3.error("Resultado: Rechazar H0")
else:
    c3.success("Resultado: No Rechazar H0")

# --- GRÁFICA DE ZONAS DE RECHAZO ---
st.write("### Gráfica de la Región Crítica")
fig3, ax3 = plt.subplots(figsize=(10, 4))
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax3.plot(x, y, color='blue', label='Distribución Z estándar')

# Sombrear zonas de rechazo
if "Bilateral" in tipo_test:
    ax3.fill_between(x, y, where=(x > z_critico) | (x < -z_critico), color='red', alpha=0.3, label='Zona de Rechazo')
elif "Derecha" in tipo_test:
    ax3.fill_between(x, y, where=(x > z_critico), color='red', alpha=0.3, label='Zona de Rechazo')
else:
    ax3.fill_between(x, y, where=(x < z_critico), color='red', alpha=0.3, label='Zona de Rechazo')

ax3.axvline(z_stat, color='black', linestyle='--', label=f'Z Calculado: {z_stat:.2f}')
ax3.legend()
st.pyplot(fig3)