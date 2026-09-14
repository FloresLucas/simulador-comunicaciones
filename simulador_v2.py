# =============================================================================
# SIMULADOR INTEGRAL DE SISTEMAS DE COMUNICACIÓN — v4.4
# Materia: Comunicaciones | Prototipo Interactivo Avanzado
# -----------------------------------------------------------------------------
# Movimiento 100% en el navegador (canvas) o en frames de Plotly: cero
# re-renders automáticos -> cero flickering. Estructura: 4 pestañas, una idea
# por pestaña, y una única barra de ajustes del sistema bajo el encabezado.
# Los canvas solo se animan cuando su pestaña es visible (rendimiento).
# =============================================================================

import json
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
html, body, p, label, button, code, input,
[data-testid="stMarkdownContainer"], .stTabs [data-baseweb="tab"] {
    font-family: 'Roboto', 'Segoe UI', system-ui, sans-serif !important;
}
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4 { font-weight: 500; letter-spacing: -0.01em; color: #e6e0e9; }
h1 { font-size: 1.8rem; margin-bottom: 0.1rem; }
div[data-testid="stMetric"] {
    background: #1d1b20;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 2px 8px rgba(0,0,0,.2);
    border: none;
}
div[data-testid="stMetric"] label { color: #9ea3b0; font-size: 13px; }
div[data-testid="stMetricValue"] { font-size: 1.9rem !important; }
.psection {
    display: flex; align-items: center; gap: 8px;
    border-left: 3px solid #8ab4f8; padding-left: 10px;
    margin: 20px 0 12px; color: #e6e0e9; font-size: 15px; font-weight: 500;
}
.psection span { font-size: 17px; }
.legend { display: inline-flex; align-items: center; gap: 6px; margin-right: 18px;
          color: #9ea3b0; font-size: 12.5px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
@media (max-width: 700px){
    div[data-testid="stHorizontalBlock"] { flex-wrap: wrap; row-gap: 0.35rem; }
    div[data-testid="stHorizontalBlock"] > div { min-width: 100% !important; flex: unset !important; }
    .block-container { padding: 0.8rem 1rem 3rem; }
    h1 { font-size: 1.45rem !important; }
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

_DEF = {
    "tipo_senal": "Onda Senoidal (Analógica)",
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

t = np.linspace(0, 1, 1000)


@st.cache_data(show_spinner=False)
def _build_cached(tipo_senal, frecuencia, amplitud, frec_muestreo, burst,
                  atenuacion_db, nivel_ruido, ancho_banda_khz, tasa_deseada_kbps, seed):
    f = frecuencia
    A = amplitud
    fs_ = int(frec_muestreo)
    senoidal = tipo_senal.startswith("Onda")
    factor = 10 ** (atenuacion_db / 20)
    sigma = nivel_ruido * (3.0 if burst else 1.0)

    def gen(ti):
        if senoidal:
            return A * np.sin(2 * np.pi * f * ti)
        return A * (np.sign(np.sin(2 * np.pi * f * ti)) + 1) / 2

    senal_original = gen(t)
    rng = np.random.default_rng(seed)
    ruido = rng.normal(0, sigma, len(t))
    senal_atenuada = senal_original * factor
    senal_recibida = senal_atenuada + ruido

    a_atenuada = A * factor
    p_tx = (A ** 2) / (2 if senoidal else 1)
    p_senal = (a_atenuada ** 2) / (2 if senoidal else 1)
    p_ruido = sigma ** 2 if nivel_ruido > 0 else 1e-10
    snr_veces = p_senal / p_ruido
    snr_db = 10 * np.log10(snr_veces)

    capacidad_bps = ancho_banda_khz * 1000 * np.log2(1 + snr_veces)
    capacidad_kbps = capacidad_bps / 1000

    Ts = 1.0 / fs_
    t_samp = np.linspace(0, 1, fs_, endpoint=False)
    muestras = gen(t_samp)
    recon_total = np.zeros_like(t)
    for k in range(fs_):
        recon_total += muestras[k] * np.sinc(fs_ * (t - k * Ts))
    pot_ref = np.mean(senal_original ** 2)
    error = 100 * np.sqrt(np.mean((senal_original - recon_total) ** 2)) / (np.sqrt(pot_ref) + 1e-12)

    return dict(
        tipo=tipo_senal, senoidal=senoidal, frecuencia=f, amplitud=A, fs=fs_, burst=burst,
        sigma=sigma, nivel_ruido=nivel_ruido,
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


# =============================================================================
# PIEZAS DE DISEÑO: intro, sección, callout, carta, veredicto, chip, encabezado
# =============================================================================
def _intro(emoji, title, text):
    st.markdown(
        f"<div style='border-left:4px solid #8ab4f8;background:#1d1b20;border-radius:10px;"
        f"padding:12px 16px;margin:2px 0 16px;'>"
        f"<b style='color:#e6e0e9;font-size:15px;'>{emoji} {title}</b>"
        f"<p style='color:#c4c7d1;font-size:12.8px;margin:5px 0 0;line-height:1.5;'>{text}</p></div>",
        unsafe_allow_html=True,
    )


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
    icono, borde, fondo = _CALLOUT[tono]
    st.markdown(
        f"<div style='display:flex;gap:10px;align-items:flex-start;"
        f"border:1px solid {borde}40;border-left:4px solid {borde};background:{fondo};"
        f"border-radius:10px;padding:11px 14px;margin:6px 0 18px;'>"
        f"<span style='font-size:16px;margin-top:1px;'>{icono}</span>"
        f"<div><div style='color:{borde};font-size:12px;font-weight:600;letter-spacing:.05em;'>{titulo.upper()}</div>"
        f"<div style='color:#c4c7d1;font-size:13px;line-height:1.5;margin-top:3px;'>{texto}</div></div></div>",
        unsafe_allow_html=True,
    )


def _card(icono, titulo, valor, detalle, color="#8ab4f8"):
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-top:3px solid {color};"
        f"border-radius:14px;padding:16px 18px;margin:6px 0;box-shadow:0 2px 10px rgba(0,0,0,.25);height:100%;'>"
        f"<div style='display:flex;align-items:center;gap:7px;color:{color};font-size:12px;"
        f"font-weight:600;letter-spacing:.06em;'><span style='font-size:15px;'>{icono}</span>{titulo.upper()}</div>"
        f"<div style='font-size:28px;font-weight:600;color:#e6e0e9;margin:9px 0 6px;font-variant-numeric:tabular-nums;'>{valor}</div>"
        f"<div style='color:#9ea3b0;font-size:12.5px;line-height:1.5;'>{detalle}</div></div>",
        unsafe_allow_html=True,
    )


def _chip(icono, etiqueta, valor, color="#9ea3b0"):
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-radius:10px;padding:9px 13px;"
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


def _chain_card(tag, icono, color, valor, filas, valor_color=None):
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;gap:10px;font-size:12.5px;'>"
        f"<span style='color:#9ea3b0;'>{k}</span><span style='color:#e6e0e9;font-variant-numeric:tabular-nums;'>{v}</span></div>"
        for k, v in filas)
    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-top:3px solid {color};"
        f"border-radius:14px;padding:16px 18px;box-shadow:0 2px 10px rgba(0,0,0,.25);height:100%;'>"
        f"<div style='display:flex;align-items:center;gap:8px;width:100%;justify-content:center;'>"
        f"<span style='font-size:19px;'>{icono}</span>"
        f"<span style='color:{color};font-size:12px;font-weight:700;letter-spacing:2px;'>{tag}</span></div>"
        f"<div style='font-size:22px;font-weight:600;color:{(valor_color or '#e6e0e9')};margin:12px 0 8px;"
        f"font-variant-numeric:tabular-nums;line-height:1.15;text-align:center;'>{valor}</div>"
        f"<div style='border-top:1px solid #2b2a2e;padding-top:9px;display:flex;flex-direction:column;gap:4px;'>{rows}</div></div>",
        unsafe_allow_html=True,
    )


def _arrow():
    st.markdown(
        "<div style='height:100%;display:flex;align-items:center;justify-content:center;'>"
        "<span style='color:#5f5f6b;font-size:26px;'>→</span></div>",
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


def _frame_slider(n, prefix):
    return [dict(
        steps=[dict(args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    label=str(k), method="animate")
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
    <span>OSCILOSCOPIO · TX → CANAL → RX · repaso ×__SPEED__</span>
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
let spare=null;
function gauss(){
  if(spare!==null){const s=spare;spare=null;return s;}
  const u=Math.random()||1e-12,v=Math.random()||1e-12;
  const m=Math.sqrt(-2*Math.log(u));
  spare=m*Math.sin(2*Math.PI*v);return m*Math.cos(2*Math.PI*v);
}
const buf=[];const BMAX=450;
const spPower=P.senoidal?Math.pow(P.A*P.factor,2)/2:Math.pow(P.A*P.factor,2)/4;
let lastD=0,frm=0;
function draw(ts){
  if(ts-lastD<33.3){requestAnimationFrame(draw);return;}
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1){requestAnimationFrame(draw);return;}
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
    const rn=gauss()*P.sigma;
    buf.push(rn);if(buf.length>BMAX)buf.shift();
    const y=toY(wave(ti)*P.factor+rn);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.stroke();ctx.shadowBlur=0;

  frm=(frm+1)%10;
  if(frm===0){
    let m1=0;
    for(const s of buf){m1+=s*s;}
    const nb=buf.length||1;
    const varN=Math.max(m1/nb,1e-14);
    const snr=10*Math.log10(spPower/varN);
    const col=snr>=15?P.ok:snr>0?P.mid:P.bad;
    badge.textContent='SNR instantáneo ≈ '+snr.toFixed(1)+' dB';
    badge.style.color=col;
  }
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
})();
</script>
"""


def _scope_component(d, speed):
    params = {
        "f": d["frecuencia"], "A": d["amplitud"], "sigma": d["sigma"],
        "factor": d["factor"], "senoidal": d["senoidal"], "speed": float(speed),
        "tx": "#7ee2b8", "rx": "#f5b04b",
        "ok": "#7ee2b8", "mid": "#f5b04b", "bad": "#e57373",
    }
    html = _SCOPE_HTML.replace("__PARAMS__", json.dumps(params)).replace("__SPEED__", f"{speed:.2f}")
    st.iframe(html, height=280, width="stretch")


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
let last=0;
function draw(ts){
  if(ts-last<125){requestAnimationFrame(draw);return;}
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1){requestAnimationFrame(draw);return;}
  last=ts;
  const NB=38,N=450,lo=-4*P.sigma,hi=4*P.sigma,range=hi-lo;
  const bs=new Float64Array(NB);
  for(let i=0;i<N;i++){
    let b=Math.floor((gauss()*P.sigma-lo)/range*NB);
    if(b<0)b=0;if(b>=NB)b=NB-1;bs[b]++;
  }
  const max=Math.max.apply(null,bs)||1;
  const bw=W/NB,base=H-4;
  ctx.clearRect(0,0,W,H);
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
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
})();
</script>
"""


def _noise_component(d):
    params = {"sigma": d["sigma"], "curve": "#f5b04b"}
    html = _NOISE_HTML.replace("__PARAMS__", json.dumps(params)).replace("__SIGMA__", f"{d['sigma']:.2f}")
    st.iframe(html, height=185, width="stretch")


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
let lastD=0;
function draw(ts){
  if(ts-lastD<33.3){requestAnimationFrame(draw);return;}
  const r=cv.getBoundingClientRect();
  if(r.width<1||r.height<1){requestAnimationFrame(draw);return;}
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
  requestAnimationFrame(draw);
}
requestAnimationFrame(draw);
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
    st.iframe(html, height=345, width="stretch")


# =============================================================================
# GRÁFICO: MUESTREO → lineal vs sinc + curva de error
# =============================================================================
def _fig_recon(d):
    fs_ = d["fs"]
    Ts = d["Ts"]
    ms = d["muestras"]
    orig = d["senal_original"]
    tsamp = d["t_samp"]

    partials = []
    acc = np.zeros_like(t)
    for k in range(fs_):
        acc = acc + ms[k] * np.sinc(fs_ * (t - k * Ts))
        partials.append(acc.copy())

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
                        subplot_titles=('Señal y reconstrucciones', 'Error de la reconstrucción sinc'))
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
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        updatemenus=_play_buttons(120),
        sliders=_frame_slider(nf, 'Muestras tomadas: '),
        margin=dict(l=40, r=40, t=85, b=40))
    return fig


# =============================================================================
# GRÁFICO: SHANNON → C(B), tu pedido R y el B mínimo necesario
# =============================================================================
def _fig_shannon(d):
    snr = d["snr_veces"]
    R = d["tasa_deseada_kbps"]
    Bv = np.arange(1, 101)
    Cv = Bv * np.log2(1 + snr)
    B = d["ancho_banda_khz"]
    bi = int(B) - 1
    Cb = Cv[bi]
    b_needed = R / np.log2(1 + snr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Bv, y=Cv, mode='lines', name='Capacidad C(B)',
                             line=dict(color='#8ab4f8', width=3),
                             fill='tozeroy', fillcolor='rgba(138,180,248,0.08)'))
    fig.add_trace(go.Scatter(x=Bv, y=np.full_like(Bv, R), mode='lines',
                             name=f'Tu pedido R = {R} kbps',
                             line=dict(color='#e57373', width=1.6, dash='dot')))

    fig.add_vline(x=B, line_dash='dash', line_color='rgba(245,176,75,0.45)', line_width=1.2)
    fig.add_annotation(
        x=B, y=Cb, showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
        arrowcolor='#f5b04b', ax=0, ay=-52,
        text=f'B = {B} kHz · C = {Cb:.1f} kbps',
        font=dict(color='#f5b04b', size=12), bgcolor='#1d1b20',
        bordercolor='#f5b04b', borderwidth=1)

    if b_needed < 1:
        nota = 'El B mínimo para tu R es menor a 1 kHz: ya alcanza'
        mark = 1
    elif b_needed > 100:
        nota = 'Ni con 100 kHz llega tu pedido R'
        mark = 100
    else:
        nota = f'B mínimo para tu pedido: ≈ {b_needed:.0f} kHz'
        mark = b_needed
    fig.add_vline(x=mark, line_dash='dash', line_color='#7ee2b8', line_width=1.6)
    fig.add_annotation(x=mark, y=0.96, yref='paper', text=nota, showarrow=False,
                       textangle=-90, font=dict(color='#7ee2b8', size=12),
                       bgcolor='rgba(20,18,24,0.7)', xanchor='right', yanchor='top')

    fig.update_layout(
        title='Capacidad del canal según el ancho de banda B (a este SNR)',
        xaxis_title='Ancho de banda B (kHz)', yaxis_title='Capacidad C (kbps)',
        template='plotly_dark', height=400, showlegend=True, hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1,
                    font=dict(size=11)),
        margin=dict(l=40, r=40, t=70, b=40))
    return fig


# =============================================================================
# ESPECTRO FFT
# =============================================================================
def _fig_spectrum(d):
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
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=55, b=40))
    return fig


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
    st.session_state["tipo_senal"] = "Tren de Pulsos (Digital)"
    st.toast("Señal cambiada a tren de pulsos.", icon="🎵")
elif _PRESET == "burst_on":
    st.session_state["burst"] = True
    st.toast("Ráfaga de ruido activa: ruido instantáneo ×3.", icon="🌩️")
elif _PRESET == "burst_off":
    st.session_state["seed"] = np.random.randint(0, 100000)
    st.session_state["burst"] = False
    st.toast("Nueva realización del ruido (otra semilla).", icon="🎲")
elif _PRESET == "surprise":
    rng = np.random.default_rng()
    st.session_state["tipo_senal"] = rng.choice(["Onda Senoidal (Analógica)", "Tren de Pulsos (Digital)"])
    st.session_state["frecuencia"] = int(rng.integers(1, 21))
    st.session_state["amplitud"] = round(float(rng.uniform(1, 10)), 1)
    st.session_state["atenuacion_db"] = round(float(rng.uniform(-20, 0)), 1)
    st.session_state["nivel_ruido"] = round(float(rng.uniform(0, 5)), 1)
    st.session_state["ancho_banda_khz"] = int(rng.integers(1, 101))
    st.session_state["tasa_deseada_kbps"] = int(rng.integers(1, 201))
    st.session_state["frec_muestreo"] = int(rng.integers(5, 101))
    st.session_state["burst"] = bool(rng.random() < 0.25)
    st.toast("Configuración aleatoria aplicada.", icon="🎰")

d0 = _build_cached(
    st.session_state["tipo_senal"],
    st.session_state["frecuencia"],
    st.session_state["amplitud"],
    st.session_state["frec_muestreo"],
    st.session_state["burst"],
    st.session_state["atenuacion_db"],
    st.session_state["nivel_ruido"],
    st.session_state["ancho_banda_khz"],
    st.session_state["tasa_deseada_kbps"],
    st.session_state["seed"],
)

# =============================================================================
# HERO + BARRA DE AJUSTES + CADENA TX → CANAL → RX
# =============================================================================
with st.sidebar:
    st.markdown("**⚙️ Ajustes del sistema**")
    st.selectbox("Tipo de señal",
                 ["Onda Senoidal (Analógica)", "Tren de Pulsos (Digital)"],
                 key="tipo_senal")
    sa1, sa2 = st.columns(2)
    with sa1:
        st.slider("Frecuencia (Hz)", min_value=1, max_value=20, step=1, key="frecuencia")
    with sa2:
        st.slider("Amplitud A (V)", min_value=1.0, max_value=10.0, step=0.5, key="amplitud")
    sb1, sb2 = st.columns(2)
    with sb1:
        st.slider("Ancho de banda B (kHz)", min_value=1, max_value=100, step=1, key="ancho_banda_khz")
    with sb2:
        st.slider("Tasa R (kbps)", min_value=1, max_value=200, step=1, key="tasa_deseada_kbps")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.slider("Desviación del ruido σ (V)", min_value=0.0, max_value=5.0, step=0.1, key="nivel_ruido")
    with sc2:
        st.slider("Atenuación (dB)", min_value=-20.0, max_value=0.0, step=0.5, key="atenuacion_db")
    st.slider("Muestreo fs (Hz)", min_value=5, max_value=100, step=1, key="frec_muestreo")
    if st.button("⚡ Variar todo al azar", type="secondary", use_container_width=True):
        st.session_state["_preset"] = "surprise"
        st.rerun()
    st.caption("Cada cambio se refleja al instante en todas las pestañas.")

st.markdown(
    "<div style='text-align:center;'>"
    "<h1 style='font-size:1.75rem;margin-bottom:.2rem;'>Simulador integral de sistemas de comunicación</h1>"
    "<div style='color:#9ea3b0;font-size:13.5px;'>Materia: Comunicaciones · prototipo interactivo para el "
    "estudio del enlace completo</div></div>",
    unsafe_allow_html=True)

st.markdown("---")

_snr = d0["snr_db"]
st.markdown(
    "<div style='text-align:center;color:#9ea3b0;font-size:12px;letter-spacing:3px;margin:14px 0 10px;'>TX → CANAL → RX</div>",
    unsafe_allow_html=True)

c1, a1, c2, a2, c3 = st.columns([4, 1, 4, 1, 4])
with c1:
    _chain_card("TX · FUENTE", "📡", "#7ee2b8",
                "Onda Senoidal" if d0["senoidal"] else "Tren de Pulsos",
                [("Frecuencia", f"{d0['frecuencia']} Hz"),
                 ("Amplitud A", f"{d0['amplitud']:.1f} V"),
                 ("Potencia TX", f"{d0['potencia_tx']:.1f} W")])
with a1:
    _arrow()
with c2:
    _chain_card("CANAL · MEDIO", "🌊", "#c4c7d1",
                f"SNR {_snr:.1f} dB",
                [("Atenuación", f"{st.session_state['atenuacion_db']:.1f} dB"),
                 ("Ruido σ", f"{d0['sigma']:.2f} V")], valor_color=_snr_color(_snr))
with a2:
    _arrow()
with c3:
    _chain_card("RX · RECEPTOR", "📻", "#f5b04b",
                f"Amplitud {d0['amplitud_atenuada']:.2f} V",
                [("Potencia RX", f"{d0['potencia_senal']:.2f} W"),
                 ("Capacidad C", f"{d0['capacidad_kbps']:.1f} kbps")])

st.markdown(
    "<div style='color:#9ea3b0;font-size:12px;margin:10px 0 2px;'>"
    "<b style='color:#7ee2b8'>TX</b> transmite la señal · el <b style='color:#c4c7d1'>CANAL</b> la atenúa y "
    "le agrega ruido gaussiano (AWGN) · <b style='color:#f5b04b'>RX</b> la recibe y mide cuántos datos caben (C)</div>",
    unsafe_allow_html=True)

st.markdown("---")

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
    _callout("proba", "Interacción entre pestañas",
             "Al incrementar σ (ruido) o la atenuación en los Ajustes del sistema, la curva RX se vuelve "
             "irregular y el SNR instantáneo (arriba a la derecha) desciende y cambia de color.")

    _section("Serie de Fourier: de qué está hecha la señal", "🎼")
    _fourier_component(d0)
    if d0["senoidal"]:
        _callout("concepto", "Una sola nota",
                 "Una senoidal es un único término de Fourier: la curva se dibuja y se repite en bucle "
                 "porque no quedan armónicas por sumar. El comportamiento completo se aprecia con el tren "
                 "de pulsos.")
        if st.button("Probar con tren de pulsos", type="secondary"):
            st.session_state["_preset"] = "pulse"
            st.rerun()
    else:
        _callout("concepto", "Suma de armónicas",
                 "Cada armónico impar que se suma acerca la forma al rectángulo. En los bordes aparecen "
                 "los sobreimpulsos de Gibbs: oscilan y nunca desaparecen.")
        _callout("ojo", "Los bordes nunca terminan de estabilizarse",
                 "El pulso requiere armónicas hasta el infinito; por eso su espectro castiga a cualquier "
                 "canal de ancho finito.")

    with st.expander("📊 Espectro FFT, TX vs RX"):
        st.plotly_chart(_fig_spectrum(d0), width="stretch")
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
        _card("🔊", "SNR del canal", f"{d0['snr_db']:.1f} dB",
              "Relación señal/ruido en dB; determina cuántos datos caben en el canal.",
              color=_snr_color(d0["snr_db"]))
    with ec2:
        _card("📡", "Potencia de señal RX", f"{d0['potencia_senal']:.2f} W",
              f"Potencia TX × factor de atenuación ({d0['factor']:.2f}×).",
              color="#f5b04b")
    with ec3:
        _card("🌊", "Potencia de ruido N", f"{d0['potencia_ruido']:.3f} W",
              f"σ² con σ = {d0['sigma']:.2f} V de ruido blanco gaussiano.",
              color="#c4c7d1")

    _callout("concepto", "Por qué el SNR gobierna el sistema",
             "El SNR se calcula como la señal recibida sobre el ruido y es el insumo de la capacidad C "
             "en la pestaña Shannon. Mejorar el enlace equivale a subir la señal o bajar el ruido.")

    _section("La distribución del ruido (AWGN)", "📉")
    _noise_component(d0)
    _callout("concepto", "La campana en progreso",
             "El ruido no tiene un valor único: cada actualización reúne 450 muestras y su histograma "
             "flota alrededor de la gaussiana teórica (línea naranja), con σ = " +
             f"{d0['sigma']:.2f} V.")

    cb1, cb2 = st.columns([1, 1])
    with cb1:
        if st.button("💥 Ráfaga de ruido ×3", use_container_width=True):
            st.session_state["_preset"] = "burst_on"
            st.rerun()
    with cb2:
        if st.button("🔁 Nueva realización del ruido", use_container_width=True):
            st.session_state["_preset"] = "burst_off"
            st.rerun()
    st.caption("La ráfaga multiplica el ruido instantáneo por 3; la nueva realización regenera la muestra "
               "de ruido con otra semilla.")

    if st.session_state["burst"]:
        _callout("ojo", "Ráfaga de ruido activa",
                 f"El ruido instantáneo se multiplicó por 3 (σ = {d0['sigma']:.2f} V). La señal puede "
                 "hundirse en tramos: es el peor escenario del canal.")
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
    nyq_ok = d0["fs"] >= d0["condicion_nyquist"]
    margen = d0["fs"] / max(d0["condicion_nyquist"], 1)
    mq1, mq2 = st.columns([2, 3])
    with mq1:
        _chip("🎚️", "Frecuencia de muestreo", f"{d0['fs']} Hz (fs)", "#8ab4f8")
    with mq2:
        if nyq_ok:
            _callout("concepto", "Condición cumplida",
                     f"fs = {d0['fs']} Hz ≥ 2f = {d0['condicion_nyquist']:.0f} Hz: con la reconstrucción "
                     "correcta, la copia reproduce la original sin error.")
        else:
            _callout("ojo", "Condición violada (aliasing)",
                     f"fs = {d0['fs']} Hz es menor que 2f = {d0['condicion_nyquist']:.0f} Hz. El espectro "
                     "se pliega sobre sí mismo y la reconstrucción ya no separa la señal de su fantasma.")
    st.progress(min(1.0, margen / 2.0),
                text=f"fs / (2f) = {margen:.2f}  →  {'cumple' if margen >= 1 else 'no cumple'} Nyquist")

    _section("Reconstrucción: sinc vs. unir los puntos", "🧪")
    st.plotly_chart(_fig_recon(d0), width="stretch")

    mr1, mr2 = st.columns([1, 2])
    with mr1:
        _card("📏", "Error con sinc (completo)", f"{d0['error_recon']:.2f} %",
              "Cerca de 0 si fs ≥ 2f; crece cuando el aliasing se introduce.",
              color="#8ab4f8")
    with mr2:
        _callout("porque", "Por qué la línea lila se queda corta y la celeste no",
                 "Unir los puntos con tramos rectos recorta los picos de la señal; la reconstrucción con "
                 "sinc, en cambio, suma una función de interpolación por muestra y reproduce la "
                 "original, siempre que fs ≥ 2f. Si fs baja de ese umbral, el área roja de error se "
                 "dispara: ese crecimiento es el aliasing.")

    if not d0["senoidal"] and nyq_ok:
        _callout("ojo", "El pulso no alcanza el 0 %",
                 "El tren de pulsos no es bandlimited: exige armónicas infinitas, por lo que su error "
                 "nunca es 0 aunque se cumpla Nyquist. Con Onda Senoidal el error es prácticamente nulo.")

# ---------------- PESTAÑA 4: SHANNON ----------------
with tab_shannon:
    _intro("⚖️ Shannon-Hartley",
           "El tope físico del canal.",
           "Todo canal tiene una capacidad máxima **C = B·log₂(1+SNR)**, en kbps. Si la tasa que se pide "
           "(R) entra en C, existe una codificación con error tan bajo como se desee; si R supera a C, es "
           "imposible. Esta pestaña confronta lo que se pide contra lo que el canal puede entregar.")

    _section("Parámetros actuales del canal", "🔢")
    ch1, ch2, ch3, ch4 = st.columns(4)
    with ch1:
        _chip("🎧", "Ancho de banda", f"{d0['ancho_banda_khz']} kHz (B)", "#8ab4f8")
    with ch2:
        _chip("📶", "SNR del canal", f"{d0['snr_db']:.1f} dB",
              _snr_color(d0["snr_db"]))
    with ch3:
        _chip("📤", "Tasa pedida", f"{d0['tasa_deseada_kbps']} kbps (R)", "#f5b04b")
    with ch4:
        _chip("📥", "Capacidad", f"{d0['capacidad_kbps']:.1f} kbps (C)", "#7ee2b8")

    _section("Veredicto: lo pedido contra lo disponible", "⚖️")
    sh_ok = d0["r_bps"] <= d0["capacidad_kbps"] * 1000
    uso = min(1.0, max(0.0, d0["r_bps"] / (d0["capacidad_kbps"] * 1000)))
    if sh_ok:
        _verdict(True, "Condición R ≤ C cumplida",
                 "La tasa pedida entra en el canal: existe una codificación con error tan bajo como se desee.")
    else:
        _verdict(False, "Condición R > C violada",
                 "Se pide más de lo que el medio permite: transmitir sin errores es imposible.")
    st.progress(uso, text=f"R ocupa el {100 * uso:.0f} % de la capacidad del canal")

    vc1, vc2 = st.columns(2)
    with vc1:
        _card("📤", "Pedís transmitir (R)", f"{d0['tasa_deseada_kbps']:.0f} kbps",
              "La velocidad que la aplicación desea mantener.",
              color="#f5b04b")
    with vc2:
        _card("📥", "El canal aguanta (C)", f"{d0['capacidad_kbps']:.1f} kbps",
              f"B·log₂(1+SNR) con B = {d0['ancho_banda_khz']} kHz y SNR = {d0['snr_db']:.1f} dB.",
              color="#8ab4f8")

    st.markdown(
        f"<div style='background:#1d1b20;border:1px solid #2b2a2e;border-radius:12px;padding:14px 18px;"
        f"text-align:center;margin:6px 0 4px;'>"
        f"<div style='color:#9ea3b0;font-size:12px;letter-spacing:.06em;margin-bottom:6px;'>LA CUENTA, CON TUS NÚMEROS</div>"
        f"<div style='font-size:19px;font-weight:600;color:#e6e0e9;'>"
        f"C = <span style='color:#8ab4f8'>{d0['ancho_banda_khz']:.0f} kHz</span> · log₂(1 + "
        f"<span style='color:#f5b04b'>{d0['snr_veces']:.1f}</span>)  ≈  "
        f"<span style='color:#7ee2b8'>{d0['capacidad_kbps']:.1f} kbps</span> "
        f"&nbsp;·&nbsp; R = {d0['tasa_deseada_kbps']:.0f} kbps</div></div>",
        unsafe_allow_html=True)

    _section("Análisis: capacidad según el ancho de banda", "📈")
    st.plotly_chart(_fig_shannon(d0), width="stretch")
    b_needed = d0["tasa_deseada_kbps"] / np.log2(1 + d0["snr_veces"])
    if b_needed < 1:
        texto_b = ("Ya alcanza: el ancho de banda mínimo para sostener su pedido es menor a 1 kHz, de "
                   f"modo que con el B actual ({d0['ancho_banda_khz']} kHz) la condición R ≤ C se cumple.")
    elif b_needed > 100:
        texto_b = (f"Ni siquiera con 100 kHz de ancho de banda se alcanzaría a sustentar R = "
                   f"{d0['tasa_deseada_kbps']} kbps. La única salida es reducir la tasa o mejorar el SNR.")
    else:
        verbo = "cumple" if sh_ok else "se viola"
        texto_b = (f"Con el SNR actual, la tasa pedida de {d0['tasa_deseada_kbps']} kbps requiere un ancho "
                   f"de banda mínimo de ≈ {b_needed:.0f} kHz (línea verde). Con el B configurado "
                   f"({d0['ancho_banda_khz']} kHz), la condición R ≤ C {verbo}.")
    _callout("porque", "Lectura del gráfico",
             f"La curva celeste es la capacidad C(B) a SNR fijo: crece con el ancho de banda, pero cada "
             f"kHz adicional rinde menos (naturaleza logarítmica). La línea roja punteada es la tasa "
             f"pedida y la vertical verde, el B mínimo que la sostiene. {texto_b}")

    _section("Escenarios de análisis", "🎬")
    sc1, sc2, sc3, sc4 = st.columns(4)
    if sc1.button("📈 Subir el ruido", use_container_width=True):
        st.session_state["_preset"] = "noise"
        st.rerun()
    if sc2.button("📉 Bajar el ancho de banda", use_container_width=True):
        st.session_state["_preset"] = "bandwidth"
        st.rerun()
    if sc3.button("🚀 Exigir más velocidad", use_container_width=True):
        st.session_state["_preset"] = "rate"
        st.rerun()
    if sc4.button("🔇 Atenuar fuerte", use_container_width=True):
        st.session_state["_preset"] = "attenuate"
        st.rerun()
    st.markdown(
        "<div style='color:#9ea3b0;font-size:12px;margin-top:8px;'>Cada escenario reconfigura el canal y "
        "todo el simulador se recalcula: la señal, el SNR y la capacidad.</div>",
        unsafe_allow_html=True)

# =============================================================================
# GUÍA DE LECTURA
# =============================================================================
with st.expander("📖 Guía de lectura — cómo leer este simulador"):
    st.markdown(
        "<div style='color:#c4c7d1;font-size:13px;margin-bottom:12px;'>Una idea por pestaña, ordenadas "
        "según el flujo físico del enlace: se genera la señal, se degrada en el canal, se muestrea y se "
        "mide su límite de información.</div>",
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
               "curva teórica. Los botones permiten aplicar una ráfaga de ruido o regenerar la muestra.")
    with gg2:
        _guide("🎯", "Muestreo",
               "Regla de oro fs ≥ 2f. Al reproducir la reconstrucción se ve que la línea celeste (sinc) "
               "acierta si la condición se cumple; el área roja (error) explota con el aliasing.")
        _guide("⚖️", "Shannon",
               "C = B·log₂(1+SNR) con los valores del sistema. El veredicto verde/rojo compara tu pedido "
               "contra el tope físico, y el gráfico señala el B mínimo que lo sustenta.")

    st.markdown(
        "<div style='color:#8ab4f8;font-size:13px;font-weight:600;margin:14px 0 6px;'>Ideas para presentar el concepto</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#c4c7d1;font-size:12.5px;line-height:1.7;'>"
        "• Al subir el ruido, el SNR cae y la capacidad C baja: menos datos caben en un canal deteriorado.<br>"
        "• Al bajar fs por debajo de 2f, la reconstrucción se desvía: el error rojo es el aliasing.<br>"
        "• En Shannon, al exigir una R ambiciosa se observa el B mínimo necesario en la línea verde.<br>"
        "• ⚡ Variar todo al azar reconfigura el sistema completo con un clic.</div>",
        unsafe_allow_html=True)