import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px # para las graficas interactivas 
from scipy import stats

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="App Estadística - Edgar Solis", layout="wide")

st.title("📊 Distribuciones y Prueba de Hipótesis")
st.markdown("""
Esta aplicación permite visualizar distribuciones y realizar pruebas de hipótesis Z 
para una muestra con varianza poblacional conocida.
""")

# --- 1. CONFIGURACIÓN DE DATOS (SIDEBAR) ---
st.sidebar.header("Configuración de la Muestra")
n = st.sidebar.slider("Tamaño de muestra (n)", 30, 500, 100)
mu_real = st.sidebar.number_input("Media real (generación)", value=100.0)
sigma_pob = st.sidebar.number_input("Desviación estándar (σ)", value=15.0)

# Persistencia de datos
if 'datos' not in st.session_state:
    st.session_state.datos = np.random.normal(loc=mu_real, scale=sigma_pob, size=n)

if st.sidebar.button("Regenerar Datos"):
    st.session_state.datos = np.random.normal(loc=mu_real, scale=sigma_pob, size=n)

df = pd.DataFrame(st.session_state.datos, columns=['Valores'])

# --- 2. VISUALIZACIÓN INTERACTIVA ---
st.header("1. Análisis Visual")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(df, x="Valores", marginal="box", 
                            title="Distribución y Boxplot",
                            color_discrete_sequence=['skyblue'])
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Resumen estadístico simple
    st.subheader("Resumen Descriptivo")
    st.write(df.describe())

# --- 3. PRUEBA DE HIPÓTESIS Z ---
st.divider()
st.header("2. Prueba de Hipótesis Z")

c_p1, c_p2 = st.columns(2)

with c_p1:
    h0_val = st.number_input("Hipótesis Nula (H0: μ = )", value=100.0)
    alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)

with c_p2:
    tipo_test = st.selectbox("Selecciona la Hipótesis Alternativa (H1)", 
                             ["Bilateral (μ ≠ H0)", "Cola Derecha (μ > H0)", "Cola Izquierda (μ < H0)"])

# --- CÁLCULOS ESTADÍSTICOS CRÍTICOS ---
media_muestral = df['Valores'].mean()
z_stat = (media_muestral - h0_val) / (sigma_pob / np.sqrt(n))

# Lógica de P-Value y Valores Críticos
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

# --- MOSTRAR RESULTADOS ---
res1, res2, res3 = st.columns(3)
res1.metric("Estadístico Z", f"{z_stat:.4f}")
res2.metric("P-Value", f"{p_value:.4f}")

if rechazar:
    res3.error("Resultado: Rechazar H0")
else:
    res3.success("Resultado: No Rechazar H0")

# --- 4. GRÁFICA DE LA CAMPANA INTERACTIVA ---
st.write("### Visualización de la Región Crítica")

x_gauss = np.linspace(-4, 4, 500)
y_gauss = stats.norm.pdf(x_gauss, 0, 1)

fig_z = go.Figure()

# Curva base
fig_z.add_trace(go.Scatter(x=x_gauss, y=y_gauss, mode='lines', name='Normal Estándar', line=dict(color='blue')))

# Sombreado dinámico de zonas de rechazo
if "Bilateral" in tipo_test:
    x_inv = np.linspace(-4, -z_critico, 100)
    fig_z.add_trace(go.Scatter(x=x_inv, y=stats.norm.pdf(x_inv), fill='tozeroy', name='Z. Rechazo Inf.', fillcolor='rgba(255,0,0,0.3)', line=dict(width=0)))
    x_sup = np.linspace(z_critico, 4, 100)
    fig_z.add_trace(go.Scatter(x=x_sup, y=stats.norm.pdf(x_sup), fill='tozeroy', name='Z. Rechazo Sup.', fillcolor='rgba(255,0,0,0.3)', line=dict(width=0)))
elif "Derecha" in tipo_test:
    x_sup = np.linspace(z_critico, 4, 100)
    fig_z.add_trace(go.Scatter(x=x_sup, y=stats.norm.pdf(x_sup), fill='tozeroy', name='Zona Rechazo', fillcolor='rgba(255,0,0,0.3)', line=dict(width=0)))
else:
    x_inv = np.linspace(-4, z_critico, 100)
    fig_z.add_trace(go.Scatter(x=x_inv, y=stats.norm.pdf(x_inv), fill='tozeroy', name='Zona Rechazo', fillcolor='rgba(255,0,0,0.3)', line=dict(width=0)))

# Línea del Z calculado
fig_z.add_vline(x=z_stat, line_width=3, line_dash="dash", line_color="black")
fig_z.add_annotation(x=z_stat, y=0.1, text=f"Tu Z: {z_stat:.2f}", showarrow=True)

fig_z.update_layout(xaxis_title="Z", yaxis_title="Probabilidad", height=400)
st.plotly_chart(fig_z, use_container_width=True)