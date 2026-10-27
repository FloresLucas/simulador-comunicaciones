# =============================================================================
# SIMULADOR INTEGRAL DE SISTEMAS DE COMUNICACIÓN — v4.7
# Materia: Comunicaciones | Prototipo Interactivo Avanzado
# -----------------------------------------------------------------------------
# Movimiento 100% en el navegador (canvas) o en frames de Plotly: cero
# re-renders automáticos -> cero flickering. Estructura: 4 pestañas, una idea
# por pestaña, y los ajustes del sistema en la barra lateral.
# Los canvas se pausan de verdad (IntersectionObserver) cuando su pestaña
# no está a la vista: el requestAnimationFrame se cancela y se reinicia.
# Shannon usa N = N₀·B con N₀ FIJO (σ² a 10 kHz): el tope P/(N₀·ln 2) no se
# mueve con el slider de B; al abrir banda el ruido crece y el SNR baja.
# El ruido de TODO el simulador (señal RX, osciloscopio, histograma) escala con
# el ancho de banda: σ_ef = σ·√(B/10 kHz). Así el SNR de las tarjetas, el del
# osciloscopio y el de Shannon son el mismo número.
# =============================================================================

import csv
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from scipy import fft as _fft
except ImportError:
    import numpy.fft as _fft

st.set_page_config(
    page_title="Simulador TFI - Comunicaciones",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS modular: todo el diseño vive en style.css (ver .clinerules).
_CSS_PATH = Path(__file__).with_name("style.css")
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True)

SEN_OLA = "Onda Senoidal (Analógica)"
SEN_PULSO = "Tren de Pulsos (Digital)"
TIPOS_SENAL = [SEN_OLA, SEN_PULSO]

_DEF = {
    "tipo_senal": SEN_OLA,
    "frecuencia": 5,
    "amplitud": 5.0,
    "atenuacion_db": -3.0,
    "nivel_ruido": 0.5,
    "ancho_banda_khz": 10,
    "tasa_deseada_kbps": 25,
    "frec_muestreo": 25,
}
for _k, _v in _DEF.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("seed", 42)
st.session_state.setdefault("burst", False)
st.session_state.setdefault("scope_speed", 0.25)
st.session_state.setdefault("modo_pres", False)
st.session_state.setdefault("base", None)

# Ventana de 1 s sin incluir el extremo: dt = 1/1000 exacto, para que la FFT
# tenga bins de 1 Hz y el pico quede en la frecuencia verdadera.
t = np.linspace(0, 1, 1000, endpoint=False)

# B de referencia del canal (kHz). N₀ se define una sola vez como σ² medida en
# esta banda: N₀ es propiedad del enlace y NO cambia cuando el usuario mueve el
# slider de B. Así el tope de Shannon P/(N₀·ln 2) queda fijo y al abrir banda
# el ruido N = N₀·B crece, el SNR baja y C(B) es cóncava.
B_REF_KHZ = 10.0

# Layout de gráficos (ver .clinerules): fondo igual al de la página y márgenes
# mínimos. El alto del margen superior se amplía solo donde hay controles.
_PLOT_BG = "#141218"
_PLOT_MARGIN = dict(l=14, r=14, t=64, b=14)


@st.cache_data(show_spinner=False)
def _build_cached(tipo_senal, frecuencia, amplitud, frec_muestreo, burst,
                  atenuacion_db, nivel_ruido, ancho_banda_khz, tasa_deseada_kbps, seed):
    f = frecuencia
    A = amplitud
    fs_ = int(frec_muestreo)
    senoidal = tipo_senal == SEN_OLA
    factor = 10 ** (atenuacion_db / 20)
    # σ_nom: desvío del ruido en la banda de referencia (10 kHz), lo que fija el
    # slider. N = N₀·B  =>  la potencia crece con B y el desvío efectivo con √B.
    sigma_nom = max(nivel_ruido, 0.0)
    escala_b = float(np.sqrt(ancho_banda_khz / B_REF_KHZ))
    sigma = sigma_nom * escala_b

    def gen(ti):
        if senoidal:
            return A * np.sin(2 * np.pi * f * ti)
        # >= 0 (y no np.sign): en los cruces por cero la muestra vale A, igual
        # que en la versión JS; np.sign(sin(0)) daría 0 y una muestra A/2 falsa.
        return A * (np.sin(2 * np.pi * f * ti) >= 0)

    senal_original = gen(t)
    rng = np.random.default_rng(seed)
    ruido_ref = rng.normal(0, sigma_nom, len(t))
    if burst:
        m = (t % 0.8) < 0.09
        ruido_ref[m] = rng.normal(0, sigma_nom * 3.0, int(m.sum()))
    ruido = ruido_ref * escala_b
    senal_atenuada = senal_original * factor
    senal_recibida = senal_atenuada + ruido

    a_atenuada = A * factor
    # Potencia media de cualquier señal 0/A o ±A con ciclo de trabajo 50 %:
    # senoidal -> A²/2 ; tren de pulsos 0/A -> A²/2 (no A² ni A²/4).
    p_tx = (A ** 2) / 2.0
    p_senal = (a_atenuada ** 2) / 2.0
    p_ruido_ref = float(np.mean(ruido_ref ** 2))   # N a B = 10 kHz
    p_ruido = float(np.mean(ruido ** 2))           # N = N₀·B a la banda elegida
    ideal = p_ruido_ref <= 0.0
    if ideal:
        snr_veces = float("inf")
        snr_db = float("inf")
        snr_ref = float("inf")
        n0 = 0.0
    else:
        # N₀ fijo por el canal: σ² medida en B de referencia (10 kHz). El ruido
        # a cualquier banda es N = N₀·B, así que el SNR de hoy es
        # P/(N₀·B): subir B aumenta el ruido y hace bajar el SNR.
        n0 = p_ruido_ref / B_REF_KHZ
        # SNR en la banda de referencia (propiedad del canal, no del slider).
        snr_ref = p_senal / p_ruido_ref
        snr_veces = p_senal / p_ruido     # = SNR_ref·B_ref/B
        snr_db = 10 * np.log10(snr_veces)

    capacidad_bps = ancho_banda_khz * 1000 * np.log2(1 + snr_veces)
    capacidad_kbps = capacidad_bps / 1000

    Ts = 1.0 / fs_
    t_samp = np.linspace(0, 1, fs_, endpoint=False)
    muestras = gen(t_samp)
    # Reconstrucción final (suma de todas las sincs). Las sumas parciales viven
    # aparte en _partials_cached: no se arrastran en este dict (hasta 100×1000
    # floats por copia) salvo que la pestaña Muestreo las pida.
    acc = np.zeros_like(t)
    for k in range(fs_):
        acc = acc + muestras[k] * np.sinc(fs_ * (t - k * Ts))
    recon_total = acc
    pot_ref = np.mean(senal_original ** 2)
    error = 100 * np.sqrt(np.mean((senal_original - recon_total) ** 2)) / (np.sqrt(pot_ref) + 1e-12)

    return dict(
        tipo=tipo_senal, senoidal=senoidal, frecuencia=f, amplitud=A, fs=fs_, burst=burst,
        sigma=sigma, sigma_nom=sigma_nom, nivel_ruido=nivel_ruido, ideal=ideal, n0=n0,
        snr_ref=snr_ref, b_ref=B_REF_KHZ,
        ancho_banda_khz=ancho_banda_khz, tasa_deseada_kbps=tasa_deseada_kbps,
        factor=factor, amplitud_atenuada=a_atenuada,
        senal_original=senal_original, senal_atenuada=senal_atenuada,
        senal_recibida=senal_recibida, ruido=ruido,
        potencia_tx=p_tx, potencia_senal=p_senal, potencia_ruido=p_ruido,
        snr_veces=snr_veces, snr_db=snr_db,
        capacidad_kbps=capacidad_kbps, r_bps=tasa_deseada_kbps * 1000,
        Ts=Ts, t_samp=t_samp, muestras=muestras,
        recon_total=recon_total, error_recon=error, condicion_nyquist=2 * f,
    )


@st.cache_data(show_spinner=False)
def _partials_cached(**P):
    """Sumas parciales de la reconstrucción sinc, cacheadas aparte.

    partials[k] es la reconstrucción con las primeras k+1 muestras. Se cachea en
    su propia función para no cargar el dict principal de _build_cached con una
    matriz de hasta 100×1000 floats en cada copia.
    """
    d = _build_cached(**P)
    fs_ = d["fs"]
    Ts = d["Ts"]
    muestras = d["muestras"]
    partials = np.empty((fs_, len(t)))
    acc = np.zeros_like(t)
    for k in range(fs_):
        acc = acc + muestras[k] * np.sinc(fs_ * (t - k * Ts))
        partials[k] = acc
    return partials


# =============================================================================
# PIEZAS DE DISEÑO: intro, sección, callout, carta, veredicto, chip, encabezado
# =============================================================================
def _intro(emoji, title, text):
    if st.session_state.get("modo_pres"):
        return
    st.markdown(
        f"<div style='border-left:4px solid #8ab4f8;background:#1d1b20;border-radius:12px;"
        f"padding:12px 16px;margin:2px 0 16px;'>"
        f"<div style='font-size:11px;letter-spacing:.09em;color:#8ab4f8;font-weight:600;"
        f"text-transform:uppercase;margin-bottom:3px;'>{emoji}</div>"
        f"<div style='color:#e6e0e9;font-size:16px;font-weight:600;'>{title}</div>"
        f"<p style='color:#c4c7d1;font-size:13px;margin:6px 0 0;line-height:1.5;'>{text}</p></div>",
        unsafe_allow_html=True,
    )


def _caption(texto):
    if not st.session_state.get("modo_pres"):
        st.caption(texto)


def _section(label, emoji=None):
    if emoji:
        st.markdown(f"<div class='psection'><span>{emoji}</span>{label}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='psection'>{label}</div>", unsafe_allow_html=True)


_CALLOUT = {
    "concepto": ("🧠", "#8ab4f8", "rgba(138,180,248,0.09)"),
    "porque": ("❓", "#f5b04b", "rgba(245,176,75,0.09)"),
    "ojo": ("⚠️", "#e57373", "rgba(229,115,115,0.10)"),
    "proba": ("🧪", "#7ee2b8", "rgba(126,226,184,0.09)"),
}


def _callout(tono, titulo, texto):
    # Modo presentación: se ocultan las explicaciones; los avisos de estado («ojo») se mantienen.
    if st.session_state.get("modo_pres") and tono != "ojo":
        return
    icono, borde, fondo = _CALLOUT[tono]
    st.markdown(
        f"<div style='display:flex;gap:10px;align-items:flex-start;"
        f"border:1px solid {borde}40;border-left:4px solid {borde};background:{fondo};"
        f"border-radius:12px;padding:11px 14px;margin:6px 0 18px;'>"
        f"<span style='font-size:16px;margin-top:1px;'>{icono}</span>"
        f"<div><div style='color:{borde};font-size:12px;font-weight:600;letter-spacing:.05em;'>{titulo.upper()}</div>"
        f"<div style='color:#c4c7d1;font-size:13px;line-height:1.5;margin-top:3px;'>{texto}</div></div></div>",
        unsafe_allow_html=True,
    )


def _card(icono, titulo, valor, detalle, color="#8ab4f8"):
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-top:3px solid {color};"
        f"border-radius:12px;padding:16px 18px;margin:6px 0;"
        f"box-shadow:0 1px 2px rgba(0,0,0,.3),0 2px 8px rgba(0,0,0,.18);height:100%;'>"
        f"<div style='display:flex;align-items:center;gap:7px;color:{color};font-size:12px;"
        f"font-weight:600;letter-spacing:.06em;'><span style='font-size:15px;'>{icono}</span>{titulo.upper()}</div>"
        f"<div style='font-size:28px;font-weight:600;color:#e6e0e9;margin:9px 0 6px;font-variant-numeric:tabular-nums;'>{valor}</div>"
        f"<div style='color:#9ea3b0;font-size:12.5px;line-height:1.5;'>{detalle}</div></div>",
        unsafe_allow_html=True,
    )


def _chip(icono, etiqueta, valor, color="#9ea3b0"):
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-radius:12px;padding:9px 13px;"
        f"display:flex;align-items:center;gap:10px;height:100%;'>"
        f"<span style='font-size:17px;'>{icono}</span>"
        f"<div><div style='color:#9ea3b0;font-size:11px;letter-spacing:.05em;'>{etiqueta.upper()}</div>"
        f"<div style='color:{color};font-size:16px;font-weight:600;font-variant-numeric:tabular-nums;'>{valor}</div></div></div>",
        unsafe_allow_html=True,
    )


def _verdict(ok, titulo, cuerpo):
    c = "#7ee2b8" if ok else "#e57373"
    bg = "rgba(126,226,184,0.10)" if ok else "rgba(229,115,115,0.12)"
    emoji = "🟢" if ok else "🔴"
    st.markdown(
        f"<div style='border:1px solid {c}55;background:{bg};border-radius:12px;padding:14px 18px;margin:4px 0 14px;'>"
        f"<div style='color:{c};font-size:12px;font-weight:700;letter-spacing:.05em;'>{emoji} {titulo.upper()}</div>"
        f"<div style='color:#e6e0e9;font-size:15px;margin-top:4px;'>{cuerpo}</div></div>",
        unsafe_allow_html=True,
    )


def _snr_color(snr):
    return "#7ee2b8" if snr >= 15 else "#f5b04b" if snr > 0 else "#e57373"


def _fmt_db(x):
    return "∞ dB" if not np.isfinite(x) else f"{x:.1f} dB"


def _fmt_kbps(x):
    return "∞ kbps" if not np.isfinite(x) else f"{x:.1f} kbps"


def _fmt_veces(x):
    if not np.isfinite(x):
        return "∞"
    if x >= 1000:
        return f"{x:.3g}"
    return f"{x:.1f}"


def _chain_card(tag, icono, color, valor, filas, valor_color=None):
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;gap:10px;font-size:13px;'>"
        f"<span style='color:#9ea3b0;'>{k}</span><span style='color:#e6e0e9;font-variant-numeric:tabular-nums;'>{v}</span></div>"
        for k, v in filas)
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-top:3px solid {color};"
        f"border-radius:12px;padding:10px 14px;"
        f"box-shadow:0 1px 2px rgba(0,0,0,.3),0 2px 8px rgba(0,0,0,.18);height:100%;'>"
        f"<div style='display:flex;align-items:center;gap:7px;width:100%;justify-content:center;'>"
        f"<span style='font-size:15px;'>{icono}</span>"
        f"<span style='color:{color};font-size:11.5px;font-weight:700;letter-spacing:1.6px;'>{tag}</span></div>"
        f"<div style='font-size:19px;font-weight:600;color:{(valor_color or '#e6e0e9')};margin:6px 0 6px;"
        f"font-variant-numeric:tabular-nums;line-height:1.15;text-align:center;'>{valor}</div>"
        f"<div style='border-top:1px solid #2b2a2e;padding-top:6px;display:flex;flex-direction:column;gap:2px;'>{rows}</div></div>",
        unsafe_allow_html=True,
    )


def _arrow():
    st.markdown(
        "<div style='height:100%;display:flex;align-items:center;justify-content:center;'>"
        "<span style='color:#5f5f6b;font-size:22px;'>→</span></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# HELPERS DE PLOTLY
# =============================================================================
def _play_buttons(duration_ms=70):
    return [dict(
        type="buttons", showactive=False, direction="left",
        x=0.0, y=1.02, xanchor="left", yanchor="bottom",
        buttons=[
            dict(label="▶ Reproducir", method="animate",
                 args=[None, {"frame": {"duration": duration_ms, "redraw": True},
                              "transition": {"duration": 0}, "fromcurrent": True, "mode": "immediate"}]),
            dict(label="⏸ Pausar", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0}, "mode": "immediate"}]),
        ])]


def _frame_slider(n, prefix, labels=None):
    # labels[m] = cantidad de muestras mostradas en el frame m (no el índice
    # del frame): el frame 0 ya muestra 1 muestra y, si fs > 32, los frames
    # saltan, así que el contador sigue al gráfico en vez de al frame.
    def _lab(k):
        return labels[k] if labels is not None else str(k)

    return [dict(
        steps=[dict(args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    label=_lab(k), method="animate")
               for k in range(n)],
        transition={"duration": 0},
        currentvalue={"visible": True, "prefix": prefix, "font": {"size": 12}},
        x=0.06, len=0.94)]


# =============================================================================
# COMPONENTE 1: OSCILOSCOPIO EN VIVO (30 fps, se pausa si la pestaña se oculta)
# =============================================================================
_SCOPE_HTML = """
<div style="border-radius:12px;overflow:hidden;background:#141218;
            box-shadow:0 1px 3px rgba(0,0,0,.35);">
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:8px 14px;font-family:'Roboto',sans-serif;font-size:12px;color:#9ea3b0;">
    <span>OSCILOSCOPIO · TX → CANAL → RX · repaso ×__SPEED__<span id="burstNote"></span></span>
    <span id="snrBadge" style="font-variant-numeric:tabular-nums;"></span>
  </div>
  <canvas id="scope" style="width:100%;height:260px;display:block;"></canvas>
</div>
<script>
(function(){
const P=__PARAMS__;
const cv=document.getElementById('scope');
const ctx=cv.getContext('2d');
const badge=document.getElementById('snrBadge');
const bnote=document.getElementById('burstNote');
if(P.burst){bnote.textContent=' · RÁFAGA';bnote.style.color='#e57373';}
let W=0,H=0;
function size(){const r=cv.getBoundingClientRect();W=r.width||1;H=r.height||1;
  const d=Math.min(window.devicePixelRatio||1,1.5);
  cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0);}
size();window.addEventListener('resize',size);
const t0=performance.now()/1000;
const TV=2.0;
function wave(ti){
  if(P.senoidal) return P.A*Math.sin(2*Math.PI*P.f*ti);
  return P.A*((Math.sin(2*Math.PI*P.f*ti)>=0?1:-1)+1)/2;
}
// Ruido determinístico en función del tiempo: la misma realización en cada
// frame, así la traza desplaza (no "hierve") y viaja con la señal.
function h1(n){const x=Math.sin(n*127.1+311.7)*43758.5453;return x-Math.floor(x);}
function noiseAt(ti){
  const q=Math.round(ti*400);
  const u=h1(q)||1e-12,v=h1(q+7919)||1e-12;
  const m=Math.sqrt(-2*Math.log(u));
  return m*Math.sin(2*Math.PI*v);
}
const buf=[];const BMAX=450;
// Potencia de la señal RX: A²/2 tanto para la senoidal como para el pulso
// 0/A al 50 % de ciclo de trabajo (igual que el cálculo en Python).
const spPower=Math.pow(P.A*P.factor,2)/2;
function burstEnv(t){const ph=(t%0.8+0.8)%0.8;return ph<0.09?1:0;}
let lastD=0,frm=0,raf=null,vis=true;
function draw(ts){
  if(!vis){raf=null;return;}
  raf=requestAnimationFrame(draw);
  if(ts-lastD<33.3)return;
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1)return;
  lastD=ts;
  const now=(performance.now()/1000-t0)*P.speed;
  ctx.clearRect(0,0,W,H);
  ctx.lineWidth=1;ctx.strokeStyle='rgba(230,224,233,0.07)';
  const sx=Math.max(26,W/10);
  for(let x=sx;x<W;x+=sx){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  const sy=Math.max(18,H/6);
  for(let y=sy;y<H;y+=sy){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  ctx.strokeStyle='rgba(230,224,233,0.16)';
  ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();

  const N=Math.max(4,Math.floor(W*1.4));
  const scale=(H-20)/2/(P.A*1.15);
  const toY=v=>H/2-v*scale;

  ctx.beginPath();ctx.strokeStyle=P.tx;ctx.lineWidth=1.5;
  for(let i=0;i<N;i++){
    const x=i/(N-1)*W;
    const ti=now-TV*(1-i/(N-1));
    const y=toY(wave(ti));
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();

  ctx.beginPath();ctx.strokeStyle=P.rx;ctx.lineWidth=2;
  ctx.shadowBlur=5;ctx.shadowColor=P.rx;
  for(let i=0;i<N;i++){
    const x=i/(N-1)*W;
    const ti=now-TV*(1-i/(N-1));
    const rn=P.sigma>0?noiseAt(ti)*P.sigma*(P.burst?1+2*burstEnv(ti):1):0;
    if(P.sigma>0){buf.push(rn);if(buf.length>BMAX)buf.shift();}
    const y=toY(wave(ti)*P.factor+rn);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();ctx.shadowBlur=0;

  frm=(frm+1)%10;
  if(frm===0){
    if(P.sigma<=0){
      badge.textContent='SNR instantáneo = ∞ (σ = 0: canal ideal)';
      badge.style.color=P.ok;
    } else {
      let m1=0;
      for(const s of buf){m1+=s*s;}
      const nb=buf.length||1;
      const varN=Math.max(m1/nb,1e-14);
      const snr=10*Math.log10(spPower/varN);
      const col=snr>=15?P.ok:snr>0?P.mid:P.bad;
      badge.textContent='SNR instantáneo ≈ '+snr.toFixed(1)+' dB';
      badge.style.color=col;
    }
  }
}
// Pausa real: si la pestaña no está a la vista se cancela el
// requestAnimationFrame y se reinicia al volver (IntersectionObserver).
const io=new IntersectionObserver(function(es){
  vis=es[0].isIntersecting;
  if(vis&&raf===null){lastD=0;raf=requestAnimationFrame(draw);}
},{threshold:0});
io.observe(cv);
raf=requestAnimationFrame(draw);
})();
</script>
"""


def _scope_component(d, speed):
    params = {
        "f": d["frecuencia"], "A": d["amplitud"], "sigma": d["sigma"],
        "factor": d["factor"], "senoidal": d["senoidal"], "burst": bool(d["burst"]),
        "speed": float(speed),
        "tx": "#7ee2b8", "rx": "#f5b04b",
        "ok": "#7ee2b8", "mid": "#f5b04b", "bad": "#e57373",
    }
    html = _SCOPE_HTML.replace("__PARAMS__", json.dumps(params)).replace("__SPEED__", f"{speed:.2f}")
    st.iframe(html, height=310, width="stretch")


# =============================================================================
# COMPONENTE 2: HISTOGRAMA AWGN EN VIVO (8 fps, pausa si la pestaña se oculta)
# =============================================================================
_NOISE_HTML = """
<div style="border-radius:12px;overflow:hidden;background:#141218;
            box-shadow:0 1px 3px rgba(0,0,0,.35);">
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:8px 14px;font-family:'Roboto',sans-serif;font-size:12px;color:#9ea3b0;">
    <span>RUIDO AWGN · distribución gaussiana en tiempo real</span>
    <span style="font-variant-numeric:tabular-nums;">σ = __SIGMA__ V</span>
  </div>
  <canvas id="hn" style="width:100%;height:150px;display:block;"></canvas>
</div>
<script>
(function(){
const P=__PARAMS__;
const cv=document.getElementById('hn');
const ctx=cv.getContext('2d');
let W=0,H=0;
function size(){const r=cv.getBoundingClientRect();W=r.width||1;H=r.height||1;
  const d=Math.min(window.devicePixelRatio||1,1.5);
  cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0);}
size();window.addEventListener('resize',size);
let spare=null;
function gauss(){
  if(spare!==null){const s=spare;spare=null;return s;}
  const u=Math.random()||1e-12,v=Math.random()||1e-12;
  const m=Math.sqrt(-2*Math.log(u));
  spare=m*Math.sin(2*Math.PI*v);return m*Math.cos(2*Math.PI*v);
}
let last=0,raf=null,vis=true;
function draw(ts){
  if(!vis){raf=null;return;}
  raf=requestAnimationFrame(draw);
  if(ts-last<125)return;
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1)return;
  last=ts;
  const NB=38,N=450;
  ctx.clearRect(0,0,W,H);
  if(!(P.sigma>0)){
    // Con σ = 0 el rango del histograma es 0 (0/0 = NaN): no hay nada que
    // dibujar, así que se informa en lugar de dejar el canvas vacío.
    ctx.fillStyle='rgba(158,163,176,1)';
    ctx.font='13px Roboto, sans-serif';
    ctx.textAlign='center';
    ctx.fillText('σ = 0: sin ruido (canal ideal)', W/2, H/2);
    return;
  }
  const lo=-4*P.sigma,hi=4*P.sigma,range=hi-lo;
  const bs=new Float64Array(NB);
  for(let i=0;i<N;i++){
    let b=Math.floor((gauss()*P.sigma-lo)/range*NB);
    if(b<0)b=0;if(b>=NB)b=NB-1;bs[b]++;
  }
  const max=Math.max.apply(null,bs)||1;
  const bw=W/NB,base=H-4;
  ctx.fillStyle='rgba(138,180,248,0.55)';
  for(let b=0;b<NB;b++){
    const bh=bs[b]/max*(H-24);
    ctx.fillRect(b*bw+1,base-bh,bw-2,bh);
  }
  ctx.strokeStyle=P.curve;ctx.lineWidth=2;ctx.beginPath();
  const sP=Math.sqrt(2*Math.PI)*P.sigma;
  for(let x=0;x<W;x+=2){
    const val=lo+(x/W)*range;
    const pdf=Math.exp(-val*val/(2*P.sigma*P.sigma))/sP;
    const y=base-pdf*sP*(H-24);
    x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();
}
const io=new IntersectionObserver(function(es){
  vis=es[0].isIntersecting;
  if(vis&&raf===null){last=0;raf=requestAnimationFrame(draw);}
},{threshold:0});
io.observe(cv);
raf=requestAnimationFrame(draw);
})();
</script>
"""


def _noise_component(d):
    params = {"sigma": d["sigma"], "curve": "#f5b04b"}
    html = _NOISE_HTML.replace("__PARAMS__", json.dumps(params)).replace("__SIGMA__", f"{d['sigma']:.2f}")
    st.iframe(html, height=192, width="stretch")


# =============================================================================
# COMPONENTE 3: SERIE DE FOURIER EN VIVO (30 fps, pausa si se oculta)
# =============================================================================
_FOUR_HTML = """
<div style="border-radius:12px;overflow:hidden;background:#141218;
            box-shadow:0 1px 3px rgba(0,0,0,.35);">
  <div style="display:flex;justify-content:space-between;align-items:center;
              padding:8px 14px;font-family:'Roboto',sans-serif;font-size:12px;color:#9ea3b0;">
    <span>__TITLE__</span>
    <span id="fbadge" style="font-variant-numeric:tabular-nums;"></span>
  </div>
  <canvas id="four" style="width:100%;height:310px;display:block;"></canvas>
</div>
<script>
(function(){
const P=__PARAMS__;
const cv=document.getElementById('four');
const ctx=cv.getContext('2d');
const badge=document.getElementById('fbadge');
let W=0,H=0;
function size(){const r=cv.getBoundingClientRect();W=r.width||1;H=r.height||1;
  const d=Math.min(window.devicePixelRatio||1,1.5);
  cv.width=W*d;cv.height=H*d;ctx.setTransform(d,0,0,d,0,0);}
size();window.addEventListener('resize',size);
const PI=Math.PI, TV=1.0;
function target(ti){
  if(P.senoidal) return P.A*Math.sin(2*PI*P.f*ti);
  return P.A*((Math.sin(2*PI*P.f*ti)>=0?1:-1)+1)/2;
}
function partial(ti,nMax){
  if(P.senoidal) return target(ti);
  let acc=P.A/2;
  for(let n=1;n<=nMax;n+=2){ acc+=P.A*(2/(PI*n))*Math.sin(2*PI*n*P.f*ti); }
  return acc;
}
const NMAX=21;
let lastD=0,raf=null,vis=true;
function draw(ts){
  if(!vis){raf=null;return;}
  raf=requestAnimationFrame(draw);
  if(ts-lastD<33.3)return;
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1)return;
  lastD=ts;
  ctx.clearRect(0,0,W,H);
  ctx.lineWidth=1;ctx.strokeStyle='rgba(230,224,233,0.07)';
  const sx=Math.max(26,W/10);
  for(let x=sx;x<W;x+=sx){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
  const sy=Math.max(18,H/6);
  for(let y=sy;y<H;y+=sy){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  ctx.strokeStyle='rgba(230,224,233,0.16)';
  ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();

  const scale=(H-20)/2/(P.A*1.15);
  const toY=v=>H/2-v*scale;
  const N=Math.max(160,Math.floor(W*1.0));
  function trace(st,color,w,dash){
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=w;
    ctx.setLineDash(dash||[]);
    for(let i=0;i<N;i++){
      const ti=(i/(N-1))*TV, x=i/(N-1)*W, y=toY(st(ti));
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();ctx.setLineDash([]);
  }
  trace(target,'rgba(230,224,233,0.30)',1.4,[6,4]);
  if(P.senoidal){
    const ph=(performance.now()/1000)%2.4;
    const K=Math.round(N*Math.min(1,ph/2.0));
    ctx.beginPath();ctx.strokeStyle=P.accent;ctx.lineWidth=3.2;
    for(let i=0;i<K;i++){
      const ti=(i/(N-1))*TV, x=i/(N-1)*W, y=toY(target(ti));
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke();
    if(K>0){
      const xx=(K-1)/(N-1)*W, yy=toY(target((K-1)/(N-1)*TV));
      ctx.beginPath();ctx.arc(xx,yy,5,0,2*PI);ctx.fillStyle='#7ee2b8';ctx.fill();
    }
    badge.textContent='1 solo término — la senoidal ES su serie de Fourier';
  } else {
    const k=Math.floor((performance.now()/1000)/0.28)%11;
    const n=2*k+1;
    trace(ti=>partial(ti,n),P.accent,3.2,[]);
    badge.textContent='Armónicas impares sumadas hasta n='+n+' (de '+NMAX+') → aproximación al pulso';
  }
}
// Pausa real: se cancela el requestAnimationFrame con la pestaña oculta.
const io=new IntersectionObserver(function(es){
  vis=es[0].isIntersecting;
  if(vis&&raf===null){lastD=0;raf=requestAnimationFrame(draw);}
},{threshold:0});
io.observe(cv);
raf=requestAnimationFrame(draw);
})();
</script>
"""


def _fourier_component(d):
    title = "SERIE DE FOURIER · suma de armónicas (se reproduce sola)"
    if d["senoidal"]:
        title = "SERIE DE FOURIER · la senoidal se dibuja (1 solo término)"
    params = {"A": d["amplitud"], "f": d["frecuencia"], "senoidal": d["senoidal"],
              "accent": "#8ab4f8"}
    html = (_FOUR_HTML.replace("__TITLE__", title)
            .replace("__PARAMS__", json.dumps(params)))
    st.iframe(html, height=360, width="stretch")


# =============================================================================
# GRÁFICO: MUESTREO → lineal vs sinc + curva de error
# =============================================================================
@st.cache_data(show_spinner=False)
def _fig_recon(P):
    d = _build_cached(**P)
    fs_ = d["fs"]
    ms = d["muestras"]
    orig = d["senal_original"]
    tsamp = d["t_samp"]
    # Las sumas parciales vienen de _partials_cached (su propio cache): acá no
    # se recalcula nada.
    partials = _partials_cached(**P)

    if fs_ > 32:
        idxs = np.unique(np.round(np.linspace(0, fs_ - 1, 32)).astype(int))
    else:
        idxs = np.arange(fs_)
    nf = len(idxs)
    rms_ref = np.sqrt(np.mean(orig ** 2)) + 1e-12

    def fdata(m):
        n = idxs[m]
        lin = np.interp(t, tsamp[:n + 1], ms[:n + 1])
        err = orig - partials[n]
        e_pct = 100.0 * np.sqrt(np.mean(err ** 2)) / rms_ref
        ytext = 0.9 * np.max(np.abs(err)) if np.any(err) else 1e-3
        return dict(
            orig=go.Scatter(x=t, y=orig, mode='lines', name='Original (TX)',
                            line=dict(color='rgba(230,224,233,0.5)', width=1.3, dash='dot')),
            muestras=go.Scatter(x=tsamp[:n + 1], y=ms[:n + 1], mode='markers', name='Muestras',
                                marker=dict(color='#f5b04b', size=8, symbol='circle')),
            lineal=go.Scatter(x=t, y=lin, mode='lines', name='Unir puntos (lineal)',
                              line=dict(color='#d98ae0', width=1.8, dash='dash')),
            sinc=go.Scatter(x=t, y=partials[n], mode='lines', name='Sinc (reconstrucción correcta)',
                            line=dict(color='#8ab4f8', width=3)),
            err_area=go.Scatter(x=t, y=err, mode='lines', name='Error (sinc vs original)',
                                line=dict(color='rgba(229,115,115,0.8)', width=1.4),
                                fill='tozeroy', fillcolor='rgba(229,115,115,0.18)'),
            err_txt=go.Scatter(x=[t[int(0.06 * len(t))]], y=[ytext], mode='text',
                               text=[f'{e_pct:.1f} %'], textfont=dict(color='#e57373', size=13),
                               textposition='top center', cliponaxis=False),
        )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.62, 0.38],
                        subplot_titles=('', 'Error de la reconstrucción sinc'))
    d0 = fdata(0)
    fig.add_trace(d0["orig"], row=1, col=1)
    fig.add_trace(d0["muestras"], row=1, col=1)
    fig.add_trace(d0["lineal"], row=1, col=1)
    fig.add_trace(d0["sinc"], row=1, col=1)
    fig.add_trace(d0["err_area"], row=2, col=1)
    fig.add_trace(d0["err_txt"], row=2, col=1)

    frames = []
    for m in range(nf):
        dm = fdata(m)
        frames.append(go.Frame(
            name=str(m),
            data=[dm["orig"], dm["muestras"], dm["lineal"], dm["sinc"], dm["err_area"], dm["err_txt"]]))
    fig.update(frames=frames)

    fig.update_yaxes(title_text='Amplitud (V)', row=1, col=1)
    fig.update_yaxes(title_text='Error (V)', row=2, col=1)
    fig.update_xaxes(title_text='Tiempo (s)', row=2, col=1)
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=12, color='#9ea3b0'))
    fig.update_layout(
        template='plotly_dark', height=520, hovermode='x unified',
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        showlegend=False,
        updatemenus=_play_buttons(120),
        sliders=_frame_slider(nf, 'Muestras tomadas: ',
                              [str(int(idxs[m]) + 1) for m in range(nf)]),
        margin=dict(_PLOT_MARGIN, t=96))
    return fig


# =============================================================================
# GRÁFICO: SHANNON → C(B), la tasa requerida R y el B mínimo necesario
# Modelo de ruido N = N₀·B con N₀ FIJO (σ² medido en B de referencia = 10 kHz):
# SNR(B) = SNR_ref·B_ref/B, así subir el slider de B no mueve la curva ni el
# tope P/(N₀·ln 2); solo corre el marcador de tu B actual y hace bajar el SNR.
# =============================================================================
def _capacidad(B_khz, snr_ref, b_ref):
    """C(B) en kbps con N = N₀·B: SNR(B) = SNR_ref·B_ref/B."""
    if not np.isfinite(snr_ref):
        return float("inf")
    return B_khz * np.log2(1.0 + snr_ref * b_ref / B_khz)


def _tope_capacidad(snr_ref, b_ref):
    """Límite de C(B) cuando B → ∞: P/(N₀·ln 2)."""
    return snr_ref * b_ref / np.log(2)


def _b_minimo(snr_ref, b_ref, R):
    """B mínimo en kHz que sostiene R (por bisección); None si R supera el tope."""
    if not np.isfinite(snr_ref):
        return 0.0
    if R >= _tope_capacidad(snr_ref, b_ref):
        return None
    lo, hi = 1e-6, max(float(b_ref), 1.0)
    while _capacidad(hi, snr_ref, b_ref) < R:
        hi *= 2
        if hi > 1e9:
            return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _capacidad(mid, snr_ref, b_ref) >= R:
            hi = mid
        else:
            lo = mid
    return hi


@st.cache_data(show_spinner=False)
def _fig_shannon(P):
    d = _build_cached(**P)
    # Curva anclada al canal (N₀ fijo): N₀ y SNR_ref no dependen del slider de B.
    snr0 = d["snr_ref"]
    b0 = d["b_ref"]
    b_act = d["ancho_banda_khz"]
    R = d["tasa_deseada_kbps"]

    if not np.isfinite(snr0):
        fig = go.Figure()
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, align="center",
            text=("Canal ideal (σ = 0): N = 0 → SNR = ∞ y C = ∞ para cualquier B > 0.<br>"
                  "No hay curva C(B) que dibujar: el canal no tiene límite."),
            font=dict(color="#7ee2b8", size=15),
            bgcolor="rgba(29,27,32,0.6)", bordercolor="#7ee2b8", borderwidth=1)
        fig.update_layout(
            title='Capacidad del canal según el ancho de banda B (ruido N = N₀·B)',
            xaxis_title='Ancho de banda B (kHz)', yaxis_title='Capacidad C (kbps)',
            template='plotly_dark', height=400, showlegend=False,
            paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(_PLOT_MARGIN, t=70))
        return fig

    Bv = np.arange(1, 101)
    Cv = _capacidad(Bv, snr0, b0)
    bi = int(b_act) - 1
    Cb = Cv[bi]
    b_needed = _b_minimo(snr0, b0, R)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Bv, y=Cv, mode='lines', name='Capacidad C(B)',
                             line=dict(color='#8ab4f8', width=3),
                             fill='tozeroy', fillcolor='rgba(138,180,248,0.08)'))
    fig.add_trace(go.Scatter(x=Bv, y=np.full_like(Bv, R), mode='lines',
                             name=f'Tasa requerida R = {R} kbps',
                             line=dict(color='#e57373', width=1.6, dash='dot')))

    fig.add_vline(x=b_act, line_dash='dash', line_color='rgba(245,176,75,0.45)', line_width=1.2)
    fig.add_trace(go.Scatter(
        x=[b_act], y=[Cb], mode='markers+text', hoverinfo='skip', showlegend=False,
        marker=dict(color='#f5b04b', size=10, line=dict(color='#141218', width=2)),
        text=[f'B = {b_act} kHz · C = {Cb:.1f} kbps'], textposition='bottom right',
        textfont=dict(color='#f5b04b', size=12)))

    if b_needed is None:
        fig.add_annotation(
            x=1, y=0.96, yref='paper', showarrow=False,
            text=f'Con este ruido R nunca entra: el tope es P/(N₀·ln 2) ≈ {_tope_capacidad(snr0, b0):.1f} kbps',
            font=dict(color='#e57373', size=12),
            bgcolor='rgba(20,18,24,0.7)', xanchor='left', yanchor='top')
    else:
        if b_needed < 1:
            nota = 'El B mínimo para tu R es menor a 1 kHz: ya alcanza'
            mark = 1
        elif b_needed > 100:
            nota = 'Ni con 100 kHz alcanza la tasa requerida'
            mark = 100
        else:
            nota = f'B mínimo para la tasa requerida: ≈ {b_needed:.0f} kHz'
            mark = b_needed
        fig.add_vline(x=mark, line_dash='dash', line_color='#7ee2b8', line_width=1.6)
        fig.add_annotation(x=mark, y=0.96, yref='paper', text=nota, showarrow=False,
                           textangle=-90, font=dict(color='#7ee2b8', size=12),
                           bgcolor='rgba(20,18,24,0.7)', xanchor='right', yanchor='top')

    fig.update_layout(
        title='Capacidad C(B) con ruido N = N₀·B (N₀ fijo: al abrir banda baja el SNR)',
        xaxis_title='Ancho de banda B (kHz)', yaxis_title='Capacidad C (kbps)',
        template='plotly_dark', height=400, showlegend=True, hovermode='x unified',
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1,
                    font=dict(size=11)),
        margin=dict(_PLOT_MARGIN, t=84))
    return fig


# =============================================================================
# ESPECTRO FFT
# =============================================================================
@st.cache_data(show_spinner=False)
def _fig_spectrum(P):
    d = _build_cached(**P)
    dt = t[1] - t[0]
    N = len(t)
    frecs = _fft.rfftfreq(N, d=dt)
    mag = np.abs(_fft.rfft(d["senal_original"])) / N
    mag[1:] *= 2
    mag_rx = np.abs(_fft.rfft(d["senal_recibida"])) / N
    mag_rx[1:] *= 2
    fmax_plot = min(max(25.0, 6.0 * d["frecuencia"]), frecs[-1])
    mask = frecs <= fmax_plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=frecs[mask], y=mag[mask], name='Espectro TX',
                         marker=dict(color='#8ab4f8'), width=0.65))
    fig.add_trace(go.Scatter(x=frecs[mask], y=mag_rx[mask], name='Espectro RX (con ruido)',
                             mode='lines', line=dict(color='rgba(229,115,115,0.85)', width=1.5)))
    harmonics = [1] if d["senoidal"] else list(range(1, 8, 2))
    for hn in harmonics:
        fh = hn * d["frecuencia"]
        if fh <= fmax_plot:
            fig.add_vline(x=fh, line_dash='dash', line_color='rgba(126,226,184,0.45)', line_width=1)
    fig.update_layout(
        title='Contenido espectral, TX vs RX',
        xaxis_title='Frecuencia (Hz)', yaxis_title='Amplitud espectral',
        template='plotly_dark', barmode='overlay', height=300,
        paper_bgcolor=_PLOT_BG, plot_bgcolor=_PLOT_BG,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
        margin=dict(_PLOT_MARGIN, t=64))
    return fig



# Casos listos para la defensa: cada uno parte de los valores por defecto y cambia
# solo lo indicado (burst y seed incluidos).
CASOS = {
    "1 · Enlace sano": {},
    "2 · Ruido alto (R > C)": dict(nivel_ruido=4.0),
    "3 · Aliasing (fs < 2f)": dict(frecuencia=10, frec_muestreo=15),
    "4 · Nyquist justo (fs = 2f)": dict(frecuencia=10, frec_muestreo=20),
    "5 · Tren de pulsos": dict(tipo_senal=SEN_PULSO, frecuencia=3, frec_muestreo=80),
    "6 · Ráfaga de ruido": dict(burst=True, nivel_ruido=1.0),
    "7 · Tasa mayor que C (abrir banda)": dict(tasa_deseada_kbps=100),
    "8 · Canal ideal": dict(nivel_ruido=0.0),
}


def _delta_html(actual, base, unidad, mejor_alto=True, dec=1):
    """Variación respecto a la base, en verde si mejora y en rojo si empeora."""
    if base is None:
        return ""
    if actual == base:
        return "<span style='color:#9ea3b0'>sin cambio</span>"
    if not (np.isfinite(actual) and np.isfinite(base)):
        return "<span style='color:#9ea3b0'>—</span>"
    d = actual - base
    if abs(d) < 0.5 * 10 ** (-dec):
        return "<span style='color:#9ea3b0'>sin cambio</span>"
    c = "#7ee2b8" if (d > 0) == mejor_alto else "#e57373"
    return f"<span style='color:{c}'>{d:+.{dec}f} {unidad}</span>"


def _base_line(clave, unidad, mejor_alto=True, dec=1):
    b = st.session_state.get("base")
    if not b:
        return ""
    return f" Δ vs base: {_delta_html(d0[clave], b[clave], unidad, mejor_alto, dec)}"


def _num(x, dec=4):
    if isinstance(x, (bool, str)):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    return "inf" if not np.isfinite(x) else round(float(x), dec)


def _csv_resultados(d):
    sess = st.session_state
    filas = [
        ("tipo_senal", d["tipo"], ""), ("frecuencia", d["frecuencia"], "Hz"),
        ("amplitud", d["amplitud"], "V"), ("atenuacion", sess["atenuacion_db"], "dB"),
        ("sigma_a_10kHz", d["sigma_nom"], "V"), ("sigma_efectivo", d["sigma"], "V"),
        ("rafaga", d["burst"], ""), ("ancho_banda", d["ancho_banda_khz"], "kHz"),
        ("tasa_requerida", d["tasa_deseada_kbps"], "kbps"), ("frec_muestreo", d["fs"], "Hz"),
        ("seed", sess["seed"], ""),
        ("potencia_tx", d["potencia_tx"], "W"), ("potencia_rx", d["potencia_senal"], "W"),
        ("potencia_ruido", d["potencia_ruido"], "W"), ("N0", d["n0"], "W/kHz"),
        ("snr", d["snr_veces"], "veces"), ("snr_db", d["snr_db"], "dB"),
        ("capacidad", d["capacidad_kbps"], "kbps"),
        ("cumple_R_menor_igual_C", d["r_bps"] <= d["capacidad_kbps"] * 1000, ""),
        ("nyquist_fs_mayor_2f", d["fs"] > d["condicion_nyquist"], ""),
        ("error_reconstruccion", d["error_recon"], "%"),
    ]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["variable", "valor", "unidad"])
    for k, v, u in filas:
        w.writerow([k, _num(v), u])
    return buf.getvalue().encode("utf-8-sig")


def _csv_senales(d):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["t_s", "tx_V", "rx_V"])
    for ti, a, b in zip(t, d["senal_original"], d["senal_recibida"]):
        w.writerow([round(float(ti), 4), round(float(a), 5), round(float(b), 5)])
    return buf.getvalue().encode("utf-8-sig")


# =============================================================================
# PRESETS PARA BOTONES DE DEMO / ESCENARIOS
# Streamlit NO permite reescribir el valor de un widget después de instanciarlo,
# así que los presets se aplican AQUÍ (antes de cualquier widget) y el botón solo
# pide un rerun.
# =============================================================================
_PRESET = st.session_state.pop("_preset", None)
if _PRESET == "noise":
    st.session_state["nivel_ruido"] = 4.0
    st.toast("Ruido elevado: σ = 4,0 V (el canal se deteriora).", icon="🌊")
elif _PRESET == "bandwidth":
    st.session_state["ancho_banda_khz"] = 3
    st.toast("Ancho de banda mínimo aplicado: B = 3 kHz.", icon="🎛️")
elif _PRESET == "rate":
    st.session_state["tasa_deseada_kbps"] = 180
    st.toast("Tasa exigida: R = 180 kbps.", icon="⚡")
elif _PRESET == "attenuate":
    st.session_state["atenuacion_db"] = -18.0
    st.toast("Canal atenuado: −18 dB.", icon="📉")
elif _PRESET == "pulse":
    st.session_state["tipo_senal"] = SEN_PULSO
    st.toast("Señal cambiada a tren de pulsos.", icon="🎵")
elif _PRESET == "burst_on":
    st.session_state["burst"] = True
    st.toast("Ráfaga de ruido temporal: interferencia fuerte que entra y sale.", icon="🌩️")
elif _PRESET == "burst_off":
    st.session_state["burst"] = False
    st.toast("Ráfaga quitada: vuelve el ruido base.", icon="🌤️")
elif _PRESET == "ideal":
    st.session_state["burst"] = False
    st.session_state["nivel_ruido"] = 0.0
    st.toast("Canal ideal: sin ruido (σ = 0). La RX replica la TX.", icon="😇")
elif _PRESET == "reseed":
    st.session_state["seed"] = int(np.random.default_rng().integers(1, 1_000_000))
    st.toast("Nueva realización de ruido generada (cambió la seed).", icon="🎲")
elif _PRESET == "reset":
    for _k, _v in _DEF.items():
        st.session_state[_k] = _v
    st.session_state["burst"] = False
    st.session_state["seed"] = 42
    st.toast("Valores por defecto restablecidos.", icon="🔄")
elif isinstance(_PRESET, str) and _PRESET.startswith("caso:"):
    _nombre = _PRESET[5:]
    for _k, _v in _DEF.items():
        st.session_state[_k] = _v
    st.session_state["burst"] = False
    st.session_state["seed"] = 42
    for _k, _v in CASOS.get(_nombre, {}).items():
        st.session_state[_k] = _v
    st.toast(f"Caso cargado: {_nombre}", icon="🎬")
elif _PRESET == "surprise":
    rng = np.random.default_rng()
    st.session_state["tipo_senal"] = str(rng.choice(TIPOS_SENAL))
    st.session_state["frecuencia"] = int(rng.integers(1, 21))
    # Round a la grilla del slider (step = 0.5): round(x*2)/2, no round(x, 1).
    st.session_state["amplitud"] = round(float(rng.uniform(1, 10)) * 2) / 2
    st.session_state["atenuacion_db"] = round(float(rng.uniform(-20, 0)) * 2) / 2
    st.session_state["nivel_ruido"] = round(float(rng.uniform(0, 5)), 1)
    st.session_state["ancho_banda_khz"] = int(rng.integers(1, 101))
    st.session_state["tasa_deseada_kbps"] = int(rng.integers(1, 201))
    st.session_state["frec_muestreo"] = int(rng.integers(5, 101))
    st.session_state["burst"] = bool(rng.random() < 0.25)
    st.toast("Configuración aleatoria aplicada.", icon="🎰")

# Parámetros escalares del sistema: son la clave de caché de las figuras.
_PARAMS = dict(
    tipo_senal=st.session_state["tipo_senal"],
    frecuencia=st.session_state["frecuencia"],
    amplitud=st.session_state["amplitud"],
    frec_muestreo=st.session_state["frec_muestreo"],
    burst=st.session_state["burst"],
    atenuacion_db=st.session_state["atenuacion_db"],
    nivel_ruido=st.session_state["nivel_ruido"],
    ancho_banda_khz=st.session_state["ancho_banda_khz"],
    tasa_deseada_kbps=st.session_state["tasa_deseada_kbps"],
    seed=st.session_state["seed"],
)
d0 = _build_cached(**_PARAMS)

# =============================================================================
# HERO + BARRA DE AJUSTES + CADENA TX → CANAL → RX
# =============================================================================
def _sb_title(txt):
    st.markdown(f"<div class='sb-title'>{txt}</div>", unsafe_allow_html=True)


def _sb_btn(col, label, preset, help_=None):
    with col:
        if st.button(label, key=f"sb_{preset}", width="stretch", help=help_):
            st.session_state["_preset"] = preset
            st.rerun()


with st.sidebar:
    st.markdown("<div class='sb-brand'>⚙️ Ajustes del sistema</div>", unsafe_allow_html=True)
    st.toggle("🎤 Modo presentación", key="modo_pres",
              help="Oculta las explicaciones largas y deja gráficos, números y avisos de estado.")

    _sb_title("Fuente · TX")
    st.selectbox("Tipo de señal", TIPOS_SENAL, key="tipo_senal")
    st.slider("Frecuencia f (Hz)", min_value=1, max_value=20, step=1, key="frecuencia")
    st.slider("Amplitud A (V)", min_value=1.0, max_value=10.0, step=0.5, key="amplitud")

    _sb_title("Canal")
    st.slider("Atenuación (dB)", min_value=-20.0, max_value=0.0, step=0.5, key="atenuacion_db")
    st.slider("Ruido σ a 10 kHz (V)", min_value=0.0, max_value=5.0, step=0.1, key="nivel_ruido",
              help="Desvío del ruido en la banda de referencia (10 kHz). Con otro ancho de banda el ruido "
                   "efectivo escala como σ·√(B/10): más banda, más ruido, menos SNR.")

    _sb_title("Enlace")
    st.slider("Ancho de banda B (kHz)", min_value=1, max_value=100, step=1, key="ancho_banda_khz")
    st.slider("Tasa requerida R (kbps)", min_value=1, max_value=200, step=1, key="tasa_deseada_kbps")

    _sb_title("Muestreo")
    st.slider("Frecuencia de muestreo fs (Hz)", min_value=5, max_value=100, step=1, key="frec_muestreo")

    _sb_title("Casos para la presentación")
    st.selectbox("Caso", list(CASOS), key="caso_sel", label_visibility="collapsed")
    if st.button("▶ Cargar caso", width="stretch", key="sb_caso"):
        st.session_state["_preset"] = "caso:" + st.session_state["caso_sel"]
        st.rerun()

    _sb_title("Escenarios · qué pasa si…")
    e1, e2 = st.columns(2)
    _sb_btn(e1, "📈 Ruido +", "noise", "Sube el ruido: σ = 4,0 V")
    _sb_btn(e2, "🔇 Atenuar", "attenuate", "Atenuación de −18 dB")
    e3, e4 = st.columns(2)
    _sb_btn(e3, "🌤️ Sin ráfaga" if st.session_state["burst"] else "🌩️ Ráfaga",
            "burst_off" if st.session_state["burst"] else "burst_on",
            "Interferencia temporal que entra y sale")
    _sb_btn(e4, "😇 Ideal", "ideal", "Canal ideal: σ = 0, sin ruido")
    e5, e6 = st.columns(2)
    _sb_btn(e5, "📉 Banda −", "bandwidth", "Baja el ancho de banda: B = 3 kHz")
    _sb_btn(e6, "🚀 Tasa +", "rate", "Sube la tasa requerida: R = 180 kbps")
    e7, e8 = st.columns(2)
    _sb_btn(e7, "🎲 Semilla", "reseed", "Nueva realización del ruido (cambia la seed)")
    _sb_btn(e8, "🔄 Reset", "reset", "Vuelve a los valores por defecto")
    if st.button("⚡ Variar todo al azar", type="secondary", width="stretch", key="sb_surprise"):
        st.session_state["_preset"] = "surprise"
        st.rerun()

    _sb_title("Comparar y exportar")
    _b = st.session_state.get("base")
    if _b:
        st.caption(f"Base fijada: SNR {_fmt_db(_b['snr_db'])} · C {_fmt_kbps(_b['capacidad_kbps'])} "
                   f"· error {_b['error_recon']:.2f} %")
        if st.button("✖ Quitar base", width="stretch", key="sb_unbase"):
            st.session_state["base"] = None
            st.rerun()
    else:
        if st.button("📌 Fijar como base", width="stretch", key="sb_base",
                     help="Guarda SNR, C y error actuales; las tarjetas muestran cuánto cambian después."):
            st.session_state["base"] = {k: d0[k] for k in ("snr_db", "capacidad_kbps", "error_recon")}
            st.rerun()
    st.download_button("⬇️ Resultados (CSV)", data=_csv_resultados(d0), file_name="simulador_resultados.csv",
                       mime="text/csv", width="stretch", key="dl_res")
    st.download_button("⬇️ Señales TX/RX (CSV)", data=_csv_senales(d0), file_name="simulador_senales.csv",
                       mime="text/csv", width="stretch", key="dl_sen")

# Siempre se emite el mismo elemento (vacío si el modo está apagado): si apareciera y
# desapareciera, Streamlit reiniciaría la pestaña activa al activar el modo.
st.markdown("<style>" + (".block-container{max-width:1500px;}" if st.session_state.get("modo_pres") else "") + "</style>",
            unsafe_allow_html=True)

st.markdown(
    "<div class='app-head'><h1>Simulador integral de sistemas de comunicación</h1>"
    "<div class='app-sub'>Comunicaciones · enlace completo TX → canal → RX</div></div>",
    unsafe_allow_html=True)

_snr = d0["snr_db"]
_B = st.session_state.get("base")
c1, a1, c2, a2, c3 = st.columns([5, 0.7, 5, 0.7, 5])
with c1:
    _chain_card("TX · FUENTE", "📡", "#7ee2b8",
                "Onda senoidal" if d0["senoidal"] else "Tren de pulsos",
                [("Frecuencia", f"{d0['frecuencia']} Hz"),
                 ("Amplitud A", f"{d0['amplitud']:.1f} V")]
                + ([("Potencia TX", f"{d0['potencia_tx']:.1f} W")] if _B else []))
with a1:
    _arrow()
with c2:
    _chain_card("CANAL", "🌊", "#c4c7d1",
                f"SNR {_fmt_db(_snr)}",
                [("Atenuación", f"{st.session_state['atenuacion_db']:.1f} dB"),
                 ("Ruido σ ef.", f"{d0['sigma']:.2f} V" + (" · ráfaga" if d0["burst"] else ""))]
                + ([("Δ SNR vs base", _delta_html(d0["snr_db"], _B["snr_db"], "dB"))] if _B else []),
                valor_color=_snr_color(_snr))
with a2:
    _arrow()
with c3:
    _chain_card("RX · RECEPTOR", "📻", "#f5b04b",
                f"{d0['amplitud_atenuada']:.2f} V",
                [("Potencia RX", f"{d0['potencia_senal']:.2f} W"),
                 ("Capacidad C", _fmt_kbps(d0["capacidad_kbps"]))]
                + ([("Δ C vs base", _delta_html(d0["capacidad_kbps"], _B["capacidad_kbps"], "kbps"))] if _B else []))

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

tab_senal, tab_canal, tab_muestreo, tab_shannon = st.tabs(
    ["🎼 Señal", "🌊 Canal", "🎯 Muestreo", "⚖️ Shannon"])

# ---------------- PESTAÑA 1: SEÑAL ----------------
with tab_senal:
    _intro("🔶 Señal",
           "El canal en vivo, de principio a fin.",
           "En esta pestaña se observa la señal transmitida (TX, en verde), la señal que llega al receptor "
           "(RX, en naranja, ya castigada por atenuación y ruido) y de qué está hecha la onda (series de "
           "Fourier). La animación corre en el navegador: la página no se recarga.")

    _section("Osciloscopio en vivo", "📟")
    spc1, spc2 = st.columns([2, 5])
    with spc1:
        st.slider("Velocidad de repaso (×)", min_value=0.05, max_value=1.0, step=0.05, key="scope_speed")
    with spc2:
        st.markdown(
            "<div style='color:#9ea3b0;font-size:12.5px;padding-top:32px;'>"
            "<span class='legend'><i style='background:#7ee2b8;'></i>TX original</span>"
            "<span class='legend'><i style='background:#f5b04b;'></i>RX recibida</span></div>",
            unsafe_allow_html=True)
    _scope_component(d0, st.session_state["scope_speed"])
    _caption("El osciloscopio genera su propio ruido en el navegador: es una realización independiente "
               "de la del modelo (seed de Python). Por eso el SNR instantáneo del recuadro puede diferir "
               "un poco del que muestra la tarjeta del canal; es el mismo proceso AWGN con muestras "
               "distintas, no un error.")
    _callout("proba", "Interacción entre pestañas",
             "Al incrementar σ (ruido) o la atenuación en los Ajustes del sistema, la curva RX se vuelve "
             "irregular y el SNR instantáneo (arriba a la derecha) desciende y cambia de color.")

    _section("Serie de Fourier: de qué está hecha la señal", "🎼")
    _fourier_component(d0)
    if d0["senoidal"]:
        _callout("concepto", "Una sola nota",
                 "Una senoidal es un único término de Fourier: la curva se dibuja y se repite en bucle "
                 "porque no quedan armónicas por sumar. El comportamiento completo se aprecia con el tren "
                 "de pulsos (se elige en «Tipo de señal», en el panel lateral).")
    else:
        _callout("concepto", "Suma de armónicas",
                 "Cada armónico impar que se suma acerca la forma al rectángulo. En los bordes aparecen "
                 "los sobreimpulsos de Gibbs: oscilan y nunca desaparecen.")
        _callout("ojo", "Los bordes nunca terminan de estabilizarse",
                 "El pulso requiere armónicas hasta el infinito; por eso su espectro castiga a cualquier "
                 "canal de ancho finito.")

    with st.expander("📊 Espectro FFT, TX vs RX"):
        st.plotly_chart(_fig_spectrum(_PARAMS), width="stretch")
        st.markdown(
            "<div style='color:#9ea3b0;font-size:12.5px;'>La senoidal concentra toda su energía en una "
            "única línea espectral. El pulso, en cambio, la reparte en armónicos impares hasta el "
            "infinito: un canal finito no puede transportarlo sin deformación.</div>",
            unsafe_allow_html=True)

# ---------------- PESTAÑA 2: CANAL ----------------
with tab_canal:
    _intro("🌊 Canal",
           "Cuánto castiga el medio.",
           "El canal no transmite la señal de forma gratuita: la atenúa y le agrega ruido gaussiano "
           "(AWGN). Los controles de atenuación y ruido están en los Ajustes del sistema; esta pestaña "
           "muestra el resultado sobre el enlace.")

    _section("Estado del enlace", "📊")
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        _card("🔊", "SNR del canal", _fmt_db(d0["snr_db"]),
              "Relación señal/ruido en dB; determina cuántos datos caben en el canal.",
              color=_snr_color(d0["snr_db"]))
    with ec2:
        _card("📡", "Potencia de señal RX", f"{d0['potencia_senal']:.2f} W",
              f"Potencia TX × factor de atenuación ({d0['factor']:.2f}×).",
              color="#f5b04b")
    with ec3:
        _card("🌊", "Potencia de ruido N", f"{d0['potencia_ruido']:.3f} W",
              ("Sin ruido: σ = 0, canal ideal (AWGN nulo)."
               if d0["ideal"] else
               f"N = N₀·B: σ efectivo = {d0['sigma']:.2f} V a B = {d0['ancho_banda_khz']} kHz "
               f"(σ = {d0['sigma_nom']:.2f} V a 10 kHz)."),
              color="#c4c7d1")

    _callout("concepto", "Por qué el SNR gobierna el sistema",
             "El SNR se calcula como la señal recibida sobre el ruido y es el insumo de la capacidad C "
             "en la pestaña Shannon. Mejorar el enlace equivale a subir la señal o bajar el ruido.")

    _section("La distribución del ruido (AWGN)", "📉")
    _noise_component(d0)
    if d0["ideal"]:
        _callout("proba", "σ = 0: sin ruido",
                 "Con el canal ideal no hay realizaciones que histogramar: el canvas lo indica y la RX "
                 "replica exactamente a la TX, sin degradación.")
    else:
        _callout("concepto", "La campana en progreso",
                 "El ruido no tiene un valor único: cada actualización reúne 450 muestras y su histograma "
                 "flota alrededor de la gaussiana teórica (línea naranja), con σ = " +
                 f"{d0['sigma']:.2f} V.")

    _caption("Los escenarios (ráfaga, canal ideal, nueva realización del ruido) están en el panel "
               "lateral, en «Escenarios».")

    if st.session_state["burst"]:
        _callout("ojo", "Ráfaga de ruido activa",
                 f"Interferencia temporal: en los tramos de ráfaga el ruido instantáneo llega a σ ≈ "
                 f"{d0['sigma'] * 3:.2f} V. La RX se hunde en picos: es el peor escenario del canal.")
    if d0["ideal"]:
        _callout("proba", "Canal ideal activo",
                 "σ = 0 ⇒ N = 0, SNR = ∞ y C = ∞: no hay pérdida de información por ruido. Es el caso "
                "límite contra el que comparar el canal degradado.")
    if d0["snr_db"] < 0:
        _callout("ojo", "SNR negativo",
                 "El ruido supera a la señal: este enlace es prácticamente inservible para transportar "
                 "información.")

# ---------------- PESTAÑA 3: MUESTREO ----------------
with tab_muestreo:
    _intro("🎯 Muestreo",
           "Guardar una señal como puntos, sin que mienta.",
           "Una señal digital se guarda como muestras. El teorema de Nyquist establece con qué frecuencia "
           "hay que tomarlas para reconstruir la original sin error — y la reconstrucción correcta no "
           "consiste en unir los puntos.")

    _section("Condición de Nyquist", "🎛️")
    fs_ = d0["fs"]
    dos_f = d0["condicion_nyquist"]
    es_senoidal = d0["senoidal"]
    # Estricto (>): con fs = 2f exacto las muestras caen en los cruces por
    # cero de la senoidal, todas valen 0 y el error es del 100 %.
    nyq_ok = fs_ > dos_f
    margen = fs_ / max(dos_f, 1)
    mq1, mq2 = st.columns([2, 3])
    with mq1:
        _chip("🎚️", "Frecuencia de muestreo", f"{fs_} Hz (fs)", "#8ab4f8")
    with mq2:
        if not es_senoidal:
            if not nyq_ok:
                _callout("ojo", "Aliasing: fs ≤ 2f (violación de Nyquist)",
                         f"fs = {fs_} Hz contra 2f = {dos_f:.0f} Hz: además de violar Nyquist, el tren "
                         "de pulsos no está limitado en banda (su ancho es infinito), así que el "
                         "espectro se pliega y la reconstrucción ni siquiera separa la señal de su "
                         "fantasma. Hace falta fs > 2f.")
            else:
                _callout("porque", "Nyquist no es suficiente para el pulso",
                         f"fs = {fs_} Hz contra 2f = {dos_f:.0f} Hz: fs > 2f es necesario, pero no "
                         "suficiente, porque el tren de pulsos no está limitado en banda (su ancho de "
                         "banda es infinito). La reconstrucción nunca llega al 0 %.")
        elif fs_ == dos_f:
            _callout("ojo", "Caso límite: fs = 2f exacto",
                     "Las muestras caen justo en los cruces por cero de la senoidal (t = 0, 1/2f, …): "
                     "todas valen 0 y la señal se pierde por completo. Hace falta fs > 2f.")
        elif nyq_ok:
            _callout("concepto", "Condición cumplida",
                     f"fs = {fs_} Hz > 2f = {dos_f:.0f} Hz: con la reconstrucción correcta, la copia "
                     "reproduce la original (salvo el efecto de borde de la suma de sincs).")
        else:
            _callout("ojo", "Condición violada (aliasing)",
                     f"fs = {fs_} Hz es menor que 2f = {dos_f:.0f} Hz. El espectro "
                     "se pliega sobre sí mismo y la reconstrucción ya no separa la señal de su fantasma.")
    if es_senoidal:
        texto_nyq = (f"fs / (2f) = {margen:.2f}  →  "
                     f"{'cumple' if nyq_ok else 'no cumple'} (se exige fs > 2f)")
    else:
        texto_nyq = (f"fs / (2f) = {margen:.2f}  →  "
                     "no cumple: fs ≤ 2f (aliasing), y encima el pulso no es limitado en banda"
                     if not nyq_ok else
                     "necesario pero no suficiente: el pulso no es limitado en banda")
    st.progress(min(1.0, margen / 2.0), text=texto_nyq)

    _section("Reconstrucción: sinc vs. unir los puntos", "🧪")
    st.plotly_chart(_fig_recon(_PARAMS), width="stretch")
    st.caption("Verde: señal original · puntos: muestras tomadas · gris: unir los puntos (lineal) · "
               "celeste: reconstrucción sinc (correcta) · rojo: error sinc vs original.")

    mr1, mr2 = st.columns([1, 2])
    with mr1:
        _card("📏", "Error con sinc (completo)", f"{d0['error_recon']:.2f} %",
              "Casi 0 con fs > 2f (el remanente es el efecto borde de la suma de sincs); crece con el "
              "aliasing cuando fs no supera 2f." + _base_line("error_recon", "%", False, 2),
              color="#8ab4f8")
    with mr2:
        _callout("porque", "Por qué la línea lila se queda corta y la celeste no",
                 "Unir los puntos con tramos rectos recorta los picos de la señal; la reconstrucción con "
                 "sinc, en cambio, suma una función de interpolación por muestra y reproduce la "
                 "original, siempre que fs > 2f. Si fs no supera ese umbral, el área roja de error se "
                 "dispara: ese crecimiento es el aliasing.")

    if not es_senoidal and nyq_ok:
        _callout("ojo", "El pulso no alcanza el 0 %",
                 "El tren de pulsos no es limitado en banda: exige armónicas infinitas, por lo que su "
                 "error nunca es 0 aunque fs > 2f. Con Onda Senoidal el error se acerca mucho a 0, pero "
                 "tampoco llega del todo.")
    elif es_senoidal and nyq_ok:
        _callout("porque", "Por qué el error nunca es exactamente 0",
                 f"La teoría da error 0 con fs > 2f, pero la suma de sincs solo abarca las muestras "
                 f"dentro de la ventana [0, 1]: en los bordes faltan las muestras anteriores y "
                 f"siguientes. Ese efecto de borde deja {d0['error_recon']:.2f} % de error acá, y se "
                 "achica al subir fs (nunca desaparece del todo).")

# ---------------- PESTAÑA 4: SHANNON ----------------
with tab_shannon:
    _intro("⚖️ Shannon-Hartley",
           "El tope físico del canal.",
           "Todo canal tiene una capacidad máxima <b>C = B·log₂(1+SNR)</b>, en kbps. Si la tasa requerida "
           "(R) entra en C, existe una codificación con error tan bajo como se desee; si R supera a C, es "
           "imposible. Esta pestaña confronta la tasa requerida con lo que el canal puede entregar.")

    _section("Parámetros actuales del canal", "🔢")
    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1:
        _chip("🎧", "Ancho de banda B", f"{d0['ancho_banda_khz']} kHz", "#8ab4f8")
    with ch2:
        _chip("📶", "SNR del canal", _fmt_db(d0["snr_db"]),
              _snr_color(d0["snr_db"]))
    with ch3:
        _chip("📤", "Tasa requerida R", f"{d0['tasa_deseada_kbps']} kbps", "#f5b04b")
    with ch4:
        _chip("📥", "Capacidad C", _fmt_kbps(d0['capacidad_kbps']), "#7ee2b8")

    _section("Veredicto: tasa requerida contra capacidad", "⚖️")
    sh_ok = d0["r_bps"] <= d0["capacidad_kbps"] * 1000
    if d0["ideal"]:
        _verdict(True, "Canal ideal: sin ruido, sin límite",
                 "Con σ = 0 el SNR es ∞ y C = ∞: cualquier tasa requerida entra en el canal sin errores.")
        st.progress(1.0, text="C = ∞ → toda la tasa requerida R entra en el canal")
    else:
        uso = min(1.0, max(0.0, d0["r_bps"] / (d0["capacidad_kbps"] * 1000)))
        if sh_ok:
            _verdict(True, "Condición R ≤ C cumplida",
                     "La tasa requerida entra en el canal: existe una codificación con error tan bajo como se desee.")
        else:
            _verdict(False, "Condición R > C violada",
                     "Se pide más de lo que el medio permite: transmitir sin errores es imposible.")
        st.progress(uso, text=f"R ocupa el {100 * uso:.0f} % de la capacidad del canal")

    vc1, vc2 = st.columns(2)
    with vc1:
        _card("📤", "Tasa requerida (R)", f"{d0['tasa_deseada_kbps']:.0f} kbps",
              "La velocidad que la aplicación necesita sostener.",
              color="#f5b04b")
    with vc2:
        if d0["ideal"]:
            _card("📥", "Capacidad del canal (C)", "∞ kbps",
                  "Canal ideal: con σ = 0 no hay ruido y la capacidad no tiene tope.",
                  color="#8ab4f8")
        else:
            _card("📥", "Capacidad del canal (C)", f"{d0['capacidad_kbps']:.1f} kbps",
                  f"B·log₂(1+SNR) con B = {d0['ancho_banda_khz']} kHz y SNR = {_fmt_db(d0['snr_db'])}."
                  + _base_line("capacidad_kbps", "kbps"),
                  color="#8ab4f8")

    if d0["ideal"]:
        cuenta = (f"C = {d0['ancho_banda_khz']:.0f} kHz · log₂(1 + ∞)  →  <span style='color:#7ee2b8'>∞ kbps</span>"
                  f"&nbsp;·&nbsp; R = {d0['tasa_deseada_kbps']:.0f} kbps (canal ideal, sin ruido)")
    else:
        cuenta = (f"C = <span style='color:#8ab4f8'>{d0['ancho_banda_khz']:.0f} kHz</span> · log₂(1 + "
                  f"<span style='color:#f5b04b'>{_fmt_veces(d0['snr_veces'])}</span>)  ≈  "
                  f"<span style='color:#7ee2b8'>{_fmt_kbps(d0['capacidad_kbps'])}</span> "
                  f"&nbsp;·&nbsp; R = {d0['tasa_deseada_kbps']:.0f} kbps")
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-radius:12px;padding:14px 18px;"
        f"text-align:center;margin:6px 0 4px;'>"
        f"<div style='color:#9ea3b0;font-size:12px;letter-spacing:.06em;margin-bottom:6px;'>LA CUENTA CON LOS VALORES ACTUALES</div>"
        f"<div style='font-size:19px;font-weight:600;color:#e6e0e9;'>"
        f"{cuenta}</div></div>",
        unsafe_allow_html=True)

    _section("Análisis: capacidad según el ancho de banda", "📈")
    st.plotly_chart(_fig_shannon(_PARAMS), width="stretch")

    if d0["ideal"]:
        _callout("concepto", "Canal ideal: no hay curva C(B)",
                 "Con σ = 0 el modelo da N = 0 para cualquier B, SNR = ∞ y C = ∞: no existe B mínimo "
                 "por calcular, pues toda tasa requerida entra sin errores. Aumentar σ en los Ajustes del "
                 "sistema permite volver a la curva C(B) con el modelo N = N₀·B.")
    else:
        _callout("concepto", "Qué modelo usa este gráfico",
                 f"El ruido del canal se modela como N = N₀·B con N₀ FIJO: "
                 f"N₀ ≈ {d0['n0'] * 1000:.1f} mW/kHz, medido una sola vez sobre el ruido real "
                 f"(σ = {d0['sigma_nom']:.2f} V) en la banda de referencia de {d0['b_ref']:.0f} kHz. "
                 "Por eso N₀ y la curva no cambian al mover el slider de B: al abrir banda el ruido "
                 "crece (N = N₀·B), el SNR baja — SNR(B) = SNR_ref·B_ref/B — y C(B) es cóncava, con "
                 f"tope fijo en ≈ {_tope_capacidad(d0['snr_ref'], d0['b_ref']):.1f} kbps "
                 "(P/(N₀·ln 2), el mismo para todo B). Nota de unidades: la señal viaja en banda base "
                 "(1–20 Hz) mientras que B mide el ancho del canal portador en kHz; son dos escalas "
                 "del mismo enlace, elegidas así para que se aprecien por separado el muestreo y la "
                 "capacidad.")
        b_needed = _b_minimo(d0["snr_ref"], d0["b_ref"], d0["tasa_deseada_kbps"])
        if b_needed is None:
            texto_b = (f"La tasa requerida de {d0['tasa_deseada_kbps']} kbps supera el tope asintótico de "
                       f"≈ {_tope_capacidad(d0['snr_ref'], d0['b_ref']):.1f} kbps "
                       "(P/(N₀·ln 2)): ni abriendo el ancho de banda al infinito entra. La única salida "
                       "es subir la señal o bajar el ruido.")
        elif b_needed < 1:
            texto_b = ("Ya alcanza: el ancho de banda mínimo para sostener la tasa requerida es menor a 1 kHz, "
                       f"de modo que con el B actual ({d0['ancho_banda_khz']} kHz) la condición R ≤ C "
                       "se cumple.")
        elif b_needed > 100:
            texto_b = (f"Ni siquiera con 100 kHz de ancho de banda se alcanzaría a sustentar R = "
                       f"{d0['tasa_deseada_kbps']} kbps. La única salida es reducir la tasa o mejorar el SNR.")
        else:
            verbo = "cumple" if sh_ok else "se viola"
            texto_b = (f"Con el SNR actual, la tasa requerida de {d0['tasa_deseada_kbps']} kbps requiere un "
                       f"ancho de banda mínimo de ≈ {b_needed:.0f} kHz (línea verde). Con el B configurado "
                       f"({d0['ancho_banda_khz']} kHz), la condición R ≤ C {verbo}.")
        _callout("porque", "Lectura del gráfico",
                 "La curva celeste es la capacidad C(B) con ruido N = N₀·B: crece con el ancho de banda, "
                 "pero cada kHz adicional rinde menos (es cóncava, no lineal). La línea roja punteada es "
                 f"la tasa requerida y la vertical verde, el B mínimo que la sostiene. {texto_b}")

# =============================================================================
# GUÍA DE LECTURA
# =============================================================================
with st.expander("📖 Guía de lectura — cómo leer este simulador"):
    st.markdown(
        "<div style='color:#c4c7d1;font-size:13px;margin-bottom:12px;'>Una idea por pestaña, ordenadas "
        "según el flujo físico del enlace: se genera la señal, se degrada en el canal, se muestrea y se "
        "mide su límite de información. En el resumen de arriba, <b style='color:#7ee2b8'>TX</b> transmite la "
        "señal, el <b style='color:#c4c7d1'>CANAL</b> la atenúa y le agrega ruido gaussiano (AWGN) y "
        "<b style='color:#f5b04b'>RX</b> la recibe y mide cuántos datos caben (C).</div>",
        unsafe_allow_html=True)

    gg1, gg2 = st.columns(2, gap="medium")
    with gg1:
        def _guide(e, ti, tx):
            st.markdown(
                f"<div style='background:#1d1b20;border-radius:12px;border:1px solid #2b2a2e;"
                f"padding:12px 14px;height:100%;'>"
                f"<div style='color:#8ab4f8;font-weight:600;font-size:13px;margin-bottom:6px;'>{e} {ti}</div>"
                f"<div style='color:#c4c7d1;font-size:12.5px;line-height:1.5;'>{tx}</div></div>",
                unsafe_allow_html=True)
        _guide("🎼", "Señal",
               "El osciloscopio corre en el navegador: la curva verde es la TX y la naranja la RX. La "
               "serie de Fourier se reproduce sola y muestra el contenido armónico de la onda.")
        _guide("🌊", "Canal",
               "Tarjetas de estado del enlace y el histograma del ruido AWGN respirando alrededor de su "
               "curva teórica. La ráfaga, el canal ideal y la nueva realización del ruido se activan desde «Escenarios» "
               "en el panel lateral.")
    with gg2:
        _guide("🎯", "Muestreo",
               "Regla de oro para la senoidal: fs > 2f (con fs = 2f exacto se pierde). Al reproducir la "
               "reconstrucción se ve que la línea celeste (sinc) acierta si la condición se cumple; el "
               "área roja (error) explota con el aliasing. El pulso no es limitado en banda: su error "
               "nunca llega a 0.")
        _guide("⚖️", "Shannon",
               "C = B·log₂(1+SNR) con los valores del sistema. El veredicto verde/rojo compara la tasa requerida "
               "contra el tope físico, y el gráfico señala el B mínimo que lo sustenta.")

    st.markdown(
        "<div style='color:#8ab4f8;font-size:13px;font-weight:600;margin:14px 0 6px;'>Ideas para presentar el concepto</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#c4c7d1;font-size:12.5px;line-height:1.7;'>"
        "• Al subir el ruido, el SNR cae y la capacidad C baja: menos datos caben en un canal deteriorado.<br>"
        "• Al bajar fs a 2f o menos, la reconstrucción se desvía: el error rojo es el aliasing.<br>"
        "• En Shannon, al exigir una R alta se observa el B mínimo necesario en la línea verde.<br>"
        "• Los botones de «Escenarios» y los «Casos para la presentación» (panel lateral) reconfiguran el sistema con un clic.<br>"
        "• «Fijar como base» guarda el estado actual: después las tarjetas muestran cuánto cambian SNR, C y error.<br>"
        "• El «Modo presentación» oculta las explicaciones largas y deja gráficos, números y avisos.</div>",
        unsafe_allow_html=True)