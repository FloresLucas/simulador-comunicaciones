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
  tren de pulsos, ráfaga de ruido, tasa mayor que C, canal ideal) y escenarios rápidos.
- **Cuatro pestañas**: Señal, Canal, Muestreo y Shannon.
- **Comparar y exportar**: fijar un caso como base, ver las diferencias (Δ) en las
  tarjetas y descargar resultados y señales en CSV.
- **Modo presentación**: ancho de página ampliado para proyectar.

## Modelo

- Señal senoidal o tren de pulsos 0/A; potencia A²/2.
- Ruido gaussiano aditivo con potencia N = N₀·B. Se usa una banda de referencia
  de 10 kHz: el desvío efectivo es σ·√(B/10).
- Muestreo con criterio estricto fs > 2f y reconstrucción por interpolación sinc.
- Capacidad de Shannon C = B·log₂(1 + SNR), con tope P/(N₀·ln 2) cuando B crece.

## Limitaciones

Modelo simplificado (solo ruido blanco gaussiano aditivo, sin cuantización ni
codificación de canal). En el tren de pulsos, el ruido dibujado por el oscilador
en JavaScript es independiente de la semilla de Python, por lo que el SNR medido
en pantalla puede variar levemente.

## Autor

Lucas Flores, Ingeniería en Sistemas de Información, UTN FRT.
