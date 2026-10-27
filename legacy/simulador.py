import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Simulador TFI - Comunicaciones", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("📡 TFI: Simulador Integral de Sistemas de Comunicación")
st.markdown("**Materia:** Comunicaciones | **Prototipo Interactivo Avanzado**")

# --- PANEL DE CONTROL (BARRA LATERAL) ---
with st.sidebar:
    st.header("⚙️ Panel de Configuración")
    
    st.markdown("---")
    st.subheader("1. Fuente y Transmisor")
    tipo_senal = st.selectbox("Tipo de Señal", ["Onda Senoidal (Analógica)", "Tren de Pulsos (Digital)"])
    frecuencia = st.slider("Frecuencia Fundamental (Hz)", min_value=1, max_value=20, value=5)
    amplitud = st.slider("Amplitud de entrada (V)", min_value=1.0, max_value=10.0, value=5.0)

    st.markdown("---")
    st.subheader("2. Medio de Transmisión (Canal)")
    atenuacion_db = st.slider("Atenuación (dB)", min_value=-20.0, max_value=0.0, value=-3.0)
    nivel_ruido = st.slider("Desviación del Ruido (V)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    
    st.markdown("---")
    st.subheader("3. Parámetros de Canal (Shannon)")
    ancho_banda_khz = st.slider("Ancho de Banda B (kHz)", min_value=1, max_value=100, value=10)
    tasa_deseada_kbps = st.slider("Tasa de Transmisión R (kbps)", min_value=1, max_value=200, value=25)

    st.markdown("---")
    st.subheader("4. Muestreo (Teorema de Nyquist)")
    frec_muestreo = st.slider("Frecuencia de Muestreo (fs en Hz)", min_value=5, max_value=100, value=25)

# --- CÁLCULOS MATEMÁTICOS ---
t = np.linspace(0, 1, 1000)

# Generación de señal
if tipo_senal == "Onda Senoidal (Analógica)":
    senal_original = amplitud * np.sin(2 * np.pi * frecuencia * t)
else:
    senal_original = amplitud * (np.sign(np.sin(2 * np.pi * frecuencia * t)) + 1) / 2

# Atenuación y Ruido
factor_atenuacion = 10 ** (atenuacion_db / 20)
amplitud_atenuada = amplitud * factor_atenuacion
senal_atenuada = senal_original * factor_atenuacion
ruido = np.random.normal(0, nivel_ruido, len(t))
senal_recibida = senal_atenuada + ruido

# Relación Señal a Ruido (SNR)
potencia_senal = (amplitud_atenuada ** 2) / (2 if tipo_senal.startswith("Onda") else 1)
potencia_ruido = nivel_ruido ** 2 if nivel_ruido > 0 else 1e-10
snr_veces = potencia_senal / potencia_ruido
snr_db = 10 * np.log10(snr_veces)

# Teorema de Shannon
B = ancho_banda_khz * 1000  
capacidad_bps = B * np.log2(1 + snr_veces)
capacidad_kbps = capacidad_bps / 1000
r_bps = tasa_deseada_kbps * 1000

# --- TABLERO DE MÉTRICAS PRINCIPALES ---
col1, col2, col3, col4 = st.columns(4)
col1.metric(label="Amplitud RX", value=f"{amplitud_atenuada:.2f} V", delta=f"{atenuacion_db} dB")
col2.metric(label="Potencia de Ruido (N)", value=f"{potencia_ruido:.2f} W")
col3.metric(label="SNR del Canal", value=f"{snr_db:.2f} dB")
col4.metric(label="Capacidad Máxima (C)", value=f"{capacidad_kbps:.2f} kbps")

st.markdown("---")

# --- GRÁFICO 1: DOMINIO DEL TIEMPO ---
fig_tiempo = go.Figure()
fig_tiempo.add_trace(go.Scatter(x=t, y=senal_original, mode='lines', name='Original (TX)', 
                                line=dict(dash='dot', color='cyan', width=2)))
fig_tiempo.add_trace(go.Scatter(x=t, y=senal_recibida, mode='lines', name='Recibida (RX)', 
                                line=dict(color='yellow', width=2),
                                fill='tozeroy', fillcolor='rgba(255, 255, 0, 0.1)'))

fig_tiempo.update_layout(
    title=f'Dominio del Tiempo - {tipo_senal}',
    xaxis_title='Tiempo (s)',
    yaxis_title='Amplitud (V)',
    template='plotly_dark',
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=40, b=40),
    height=350
)
st.plotly_chart(fig_tiempo, width="stretch")

# --- GRÁFICO 2: ANÁLISIS 3D DE ARMÓNICAS ---
st.subheader("🌌 Análisis 3D: Espectro y Armónicas (Serie de Fourier)")
fig_3d = go.Figure()

armonicas = [
    {"frec_mult": 1, "amp_factor": 1.0, "color": "cyan", "nombre": "Fundamental (f0)"},
    {"frec_mult": 3, "amp_factor": 1.0/3 if tipo_senal != "Onda Senoidal (Analógica)" else 0, "color": "magenta", "nombre": "3ra Armónica"},
    {"frec_mult": 5, "amp_factor": 1.0/5 if tipo_senal != "Onda Senoidal (Analógica)" else 0, "color": "yellow", "nombre": "5ta Armónica"}
]

for armonica in armonicas:
    if armonica["amp_factor"] > 0:
        f_armcnica = frecuencia * armonica["frec_mult"]
        a_armonica = (amplitud * factor_atenuacion) * armonica["amp_factor"]
        onda_armonica = a_armonica * np.sin(2 * np.pi * f_armcnica * t)
        
        fig_3d.add_trace(go.Scatter3d(
            x=t, y=onda_armonica, z=np.full(len(t), f_armcnica),
            mode='lines', name=armonica["nombre"],
            line=dict(color=armonica["color"], width=4)
        ))

fig_3d.update_layout(
    title='Superposición Espacial de Frecuencias (Rotable en 3D)',
    scene=dict(
        xaxis_title='Tiempo (s)',
        yaxis_title='Amplitud (V)',
        zaxis_title='Frecuencia (Hz)',
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5))
    ),
    template='plotly_dark',
    margin=dict(l=0, r=0, t=40, b=0),
    height=400
)
st.plotly_chart(fig_3d, width="stretch")

# --- MÓDULO VISUAL: TEOREMA DE NYQUIST ---
st.markdown("---")
st.subheader("⚡ Módulo Dinámico: Muestreo y Teorema de Nyquist")

t_muestras = np.linspace(0, 1, frec_muestreo)
if tipo_senal == "Onda Senoidal (Analógica)":
    muestras = amplitud * np.sin(2 * np.pi * frecuencia * t_muestras)
else:
    muestras = amplitud * (np.sign(np.sin(2 * np.pi * frecuencia * t_muestras)) + 1) / 2

fig_nyquist = go.Figure()
fig_nyquist.add_trace(go.Scatter(x=t, y=senal_original, mode='lines', name='Señal Continua Original', 
                                 line=dict(color='rgba(0, 255, 255, 0.4)', width=1.5)))
fig_nyquist.add_trace(go.Scatter(x=t_muestras, y=muestras, mode='markers', name='Muestras Tomadas', 
                                 marker=dict(color='orange', size=10, symbol='circle')))

fig_nyquist.update_layout(
    title='Visualización del Proceso de Muestreo Discreto',
    xaxis_title='Tiempo (s)',
    yaxis_title='Amplitud (V)',
    template='plotly_dark',
    height=350,
    margin=dict(l=40, r=40, t=40, b=40)
)
st.plotly_chart(fig_nyquist, width="stretch")

condicion_nyquist = 2 * frecuencia
col_nyq1, col_nyq2 = st.columns(2)
with col_nyq1:
    st.write(f"""
    * **Frecuencia Máxima de la Señal ($f$):** {frecuencia} Hz
    * **Condición de Nyquist ($f_s \\ge 2f$):** Se requieren al menos **{condicion_nyquist} Hz**
    * **Frecuencia Actual Configurada ($f_s$):** {frec_muestreo} Hz
    """)
with col_nyq2:
    if frec_muestreo >= condicion_nyquist:
        st.success("🟢 **Muestreo Óptimo:** Se cumple el Teorema de Nyquist. La señal puede ser reconstruida perfectamente sin pérdida de información.")
    else:
        st.error("🔴 **Peligro de Aliasing:** La frecuencia de muestreo es inferior al doble de la frecuencia máxima. La señal sufrirá distorsión y no podrá reconstruirse.")

# --- EVALUACIÓN DEL SEGUNDO TEOREMA DE SHANNON ---
st.markdown("---")
st.subheader("⚖️ Evaluación del Segundo Teorema de Shannon")

col_shannon1, col_shannon2 = st.columns([2, 1])

with col_shannon1:
    st.write(f"""
    * **Tasa de Transmisión Deseada ($R$):** {tasa_deseada_kbps} kbps
    * **Capacidad Límite del Canal ($C$):** {capacidad_kbps:.2f} kbps
    """)
    
    if r_bps <= capacidad_bps:
        st.success("🟢 **Condición $R \\le C$ Cumplida:** Existe una técnica de codificación tal que la fuente puede transmitir sobre el canal con una frecuencia de errores arbitrariamente pequeña, no obstante la presencia de ruido[cite: 1].")
    else:
        st.error("🔴 **Condición $R > C$ Violada:** No es posible transmitir sin errores con esta tasa de datos bajo las condiciones actuales de ruido y ancho de banda[cite: 1].")

with col_shannon2:
    st.info("💡 **Tip para la presentación:** Muestra cómo al bajar de golpe la frecuencia de muestreo por debajo del doble de la frecuencia de la señal, los puntos naranjas se desacoplan y salta la alerta de *Aliasing*.")
    