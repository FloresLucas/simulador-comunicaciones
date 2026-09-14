# Simulador integral de sistemas de comunicación

Prototipo interactivo para la materia Comunicaciones. Simula el enlace
completo: generación de la señal, degradación por canal (atenuación + ruido
AWGN), muestreo/reconstrucción y el límite de Shannon-Hartley.

## Pestañas

- **Señal**: osciloscopio en vivo (TX → CANAL → RX) y serie de Fourier animada, todo en canvas del navegador.
- **Canal**: estado del enlace y histograma AWGN en tiempo real.
- **Muestreo**: condición de Nyquist (fs ≥ 2f) y reconstrucción sinc vs. unir puntos.
- **Shannon**: capacidad C = B·log₂(1+SNR) contra la tasa pedida, con veredicto y B mínimo.

## Correrlo localmente

```bash
pip install -r requirements.txt
python -m streamlit run simulador_v2.py
```

## Desplegar (Streamlit Community Cloud)

1. Subir este repositorio a GitHub.
2. En https://share.streamlit.io elegir "Create app" → repositorio + rama → `simulador_v2.py`.
3. Obtener la URL pública; la aplicación no requiere secretos.

## Móvil

La interfaz apila las columnas en pantallas angostas. Para verla sin internet
se puede ejecutar el mismo archivo en Termux (Android):

```bash
pkg install python
pip install streamlit numpy plotly scipy
python -m streamlit run simulador_v2.py
```

y abrir `http://localhost:8501` en el navegador del celular.