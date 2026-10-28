# Simulador integral de sistemas de comunicación

Aplicación Streamlit para la materia Comunicaciones (UTN FRT). Muestra el enlace
TX → canal → RX y permite ver en vivo los límites de **Nyquist** (muestreo) y
**Shannon** (capacidad del canal).

## Ejecutar

```bash
pip install -r requirements.txt
streamlit run simulador_v2.py
```

`style.css` debe estar en la misma carpeta que `simulador_v2.py`.

## Qué incluye

- **Barra lateral** con controles agrupados: Fuente · TX, Canal, Enlace y Muestreo.
- **Casos** listos para cargar (enlace sano, ruido alto, aliasing, Nyquist justo,
  tren de pulsos, ráfaga de ruido, tasa mayor que C, canal ideal, interferencia de tono,
  ruido impulsivo y desvanecimiento) y escenarios rápidos.
- **Perturbaciones del canal** además del AWGN: interferencia de tono, ruido impulsivo y desvanecimiento, con su espectro, histograma y relación señal/interferencia (pestaña Canal).
- **Cinco pestañas**: Señal, Canal, Muestreo, Shannon y Digital.
- **Digital**: cuantización con N bits (regla de 6 dB por bit) y 12 modulaciones digitales (OOK, 4-ASK, 2-FSK, BPSK, DPSK, QPSK, 8-PSK, 16-PSK, 8-QAM, 16-QAM, 32-QAM y 64-QAM). Para cada una: la portadora modulada, la constelación recibida, la BER teórica, el error medido por Monte Carlo y su costo en Eb/N0 frente al límite de Shannon. Eb/N0 = SNR·B/R conecta esta pestaña con el resto.
- **Comparar y exportar**: fijar un caso como base, ver las diferencias (Δ) en las
  tarjetas y descargar resultados y señales en CSV.
- **Modo presentación**: ancho de página ampliado para proyectar.

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

## Modelo

- Señal senoidal o tren de pulsos 0/A; potencia A²/2.
- Ruido gaussiano aditivo con potencia N = N₀·B. Se usa una banda de referencia
  de 10 kHz: el desvío efectivo es σ·√(B/10).
- Muestreo con criterio estricto fs > 2f y reconstrucción por interpolación sinc.
- Cuantizador uniforme de N bits sobre el rango de la señal.
- BER: BPSK/QPSK Q(√(2·Eb/N0)); DPSK ½·e^(−Eb/N0); OOK y 2-FSK Q(√(Eb/N0)); M-arias con codificación Gray (BER ≈ error de símbolo / k), con fórmula exacta para ASK, QPSK y QAM cuadrada y cota de vecinos más cercanos para 8-PSK, 16-PSK, 8-QAM y 32-QAM.
- Capacidad de Shannon C = B·log₂(1 + SNR), con tope P/(N₀·ln 2) cuando B crece.

## Limitaciones

Las perturbaciones adicionales se tratan como parte del ruido N = N₀·B (se escalan con el ancho de banda) para mantener un único modelo; el osciloscopio de la pestaña Señal solo dibuja el AWGN y la ráfaga.


Modelo simplificado (solo ruido blanco gaussiano aditivo, sin codificación de canal
ni sincronización). En el tren de pulsos, el ruido dibujado por el oscilador
en JavaScript es independiente de la semilla de Python, por lo que el SNR medido
en pantalla puede variar levemente.

## Autor

Lucas Flores, Ingeniería en Sistemas de Información, UTN FRT.
