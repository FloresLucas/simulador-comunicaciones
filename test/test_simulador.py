"""Tests del simulador con el framework de pruebas de Streamlit (AppTest).

Se ejecutan desde la carpeta del proyecto:   pytest tests -q
Requieren:  pip install pytest
"""
import math
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "simulador_v2.py")
PULSO = "Tren de Pulsos (Digital)"


def run(**estado):
    at = AppTest.from_file(APP, default_timeout=120)
    for k, v in estado.items():
        at.session_state[k] = v
    at.run()
    return at


def md(at):
    return " ".join(m.value for m in at.markdown)


def buscar(at, patron):
    m = re.search(patron, md(at))
    return m.group(1) if m else None


def chip(at, etiqueta):
    return buscar(at, rf"{etiqueta}</div><div[^>]*>([^<]*)<")


def sin_error(at):
    assert not at.exception, [e.value for e in at.exception]


# ---------------------------------------------------------------- general
def test_arranca_con_cinco_pestanas():
    at = run()
    sin_error(at)
    assert len(at.tabs) == 5


@pytest.mark.parametrize("estado", [
    {},
    dict(nivel_ruido=0.0),
    dict(tipo_senal=PULSO),
    dict(ancho_banda_khz=100, nivel_ruido=5.0),
    dict(modo_pres=True),
    dict(modo_pres=True, nivel_ruido=0.0),
    dict(nivel_ruido=5.0, tasa_deseada_kbps=200, ancho_banda_khz=100),
])
def test_escenarios_sin_excepciones(estado):
    sin_error(run(**estado))


# ------------------------------------------------- SNR coherente con B
@pytest.mark.parametrize("B", [1, 10, 50, 100])
def test_snr_de_la_tarjeta_coincide_con_la_teoria(B):
    at = run(ancho_banda_khz=B)
    sin_error(at)
    snr = float(buscar(at, r"SNR (-?[0-9.]+) dB"))
    potencia = (5 * 10 ** (-3 / 20)) ** 2 / 2
    teorico = 10 * math.log10(potencia / (0.25 * B / 10))
    assert abs(snr - teorico) < 0.6          # el ruido es una realización finita
    sigma = float(buscar(at, r"Ruido σ ef\.</span><span[^>]*>([0-9.]+) V"))
    assert abs(sigma - 0.5 * math.sqrt(B / 10)) < 0.01


# ---------------------------------------------------------------- casos
def test_todos_los_casos_cargan():
    sel = [s for s in run().selectbox if s.key == "caso_sel"][0]
    for nombre in list(sel.options):
        at = run()
        [s for s in at.selectbox if s.key == "caso_sel"][0].select(nombre).run()
        [b for b in at.button if b.label == "▶ Cargar caso"][0].click().run()
        sin_error(at)


# --------------------------------------------------------- base y deltas
def test_fijar_y_quitar_base():
    at = run()
    [b for b in at.button if b.label == "📌 Fijar como base"][0].click().run()
    assert at.session_state["base"] is not None
    at.session_state["nivel_ruido"] = 2.0
    at.run()
    sin_error(at)
    assert "Δ SNR vs base" in md(at)
    [b for b in at.button if b.label == "✖ Quitar base"][0].click().run()
    assert at.session_state["base"] is None


def test_todos_los_botones_de_la_barra_lateral():
    for etiqueta in [b.label for b in run().button]:
        at = run()
        [b for b in at.button if b.label == etiqueta][0].click().run()
        sin_error(at)


def test_botones_de_descarga():
    assert len(run().get("download_button")) == 2


# ------------------------------------------------- pestaña Digital
@pytest.mark.parametrize("bits", [1, 4, 8, 12])
def test_sqnr_sigue_la_regla_de_6_db_por_bit(bits):
    at = run(bits_q=bits)
    sin_error(at)
    sqnr = float(chip(at, "SQNR MEDIDA").replace(" dB", ""))
    assert abs(sqnr - (6.02 * bits + 1.76)) < 2.0


TODAS = ["OOK", "4-ASK", "2-FSK", "BPSK", "DPSK", "QPSK", "8-PSK", "16-PSK",
         "8-QAM", "16-QAM", "32-QAM", "64-QAM"]


def _ser(at):
    teorico = float(chip(at, "ERROR DE SÍMBOLO TEÓRICO"))
    medido = float(buscar(at, r"MEDIDO[^<]*</div><div[^>]*>([^<]*)<").split()[0])
    return teorico, medido


@pytest.mark.parametrize("mod", TODAS)
def test_error_de_simbolo_medido_coincide_con_la_teoria(mod):
    # Se busca un Eb/N0 donde el error es medible (2 % a 8 %) y se compara con la teoría.
    for ebn0 in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0):
        at = run(mod_sel=mod, usar_enlace=False, ebn0_manual=ebn0)
        sin_error(at)
        teorico, medido = _ser(at)
        if teorico < 0.08:
            break
    assert 0.0 < teorico < 0.08
    assert abs(medido - teorico) < 0.25 * teorico + 0.002


@pytest.mark.parametrize("mod", TODAS)
def test_modulaciones_en_todos_los_modos(mod):
    for estado in ({}, dict(nivel_ruido=0.0), dict(modo_pres=True), dict(tipo_perturb="Ruido impulsivo")):
        sin_error(run(mod_sel=mod, **estado))


def test_ber_de_bpsk_y_qpsk_es_igual_y_dpsk_es_peor():
    ber = {m: float(chip(run(mod_sel=m, usar_enlace=False, ebn0_manual=8.0), "BER TEÓRICA"))
           for m in ("BPSK", "QPSK", "DPSK", "16-QAM")}
    assert ber["BPSK"] == ber["QPSK"]
    assert ber["DPSK"] > ber["BPSK"]
    assert ber["16-QAM"] > ber["BPSK"]


def test_canal_ideal_no_tiene_errores_de_bit():
    at = run(nivel_ruido=0.0)
    sin_error(at)
    assert chip(at, "BER TEÓRICA") == "0"


# ------------------------------------------- perturbaciones del canal
TONO, IMPULSO, FADING = "Interferencia de tono", "Ruido impulsivo", "Desvanecimiento"


def snr_db(at):
    return float(buscar(at, r"SNR (-?[0-9.]+) dB"))


@pytest.mark.parametrize("tipo", [TONO, IMPULSO, FADING])
@pytest.mark.parametrize("estado", [
    {}, dict(tipo_senal=PULSO), dict(nivel_ruido=0.0), dict(ancho_banda_khz=100), dict(modo_pres=True),
])
def test_perturbaciones_sin_excepciones(tipo, estado):
    sin_error(run(tipo_perturb=tipo, intens_perturb=70, **estado))


@pytest.mark.parametrize("tipo", [TONO, IMPULSO])
def test_perturbacion_aditiva_baja_el_snr(tipo):
    base = snr_db(run())
    con = snr_db(run(tipo_perturb=tipo, intens_perturb=60))
    assert con < base - 2


def test_tono_cumple_la_relacion_senal_interferencia():
    at = run(tipo_perturb=TONO, intens_perturb=50)
    sir = float(chip(at, r"SEÑAL/INTERFERENCIA \(SIR\)").replace(" dB", ""))
    assert abs(sir - 20 * math.log10(1 / 0.5)) < 0.3      # la amplitud es el 50 % de la señal


def test_con_ninguna_el_resultado_no_cambia():
    assert snr_db(run(tipo_perturb="Ninguna (solo AWGN)", intens_perturb=90)) == snr_db(run())


def test_el_boton_ideal_quita_la_perturbacion():
    at = run(tipo_perturb=TONO, intens_perturb=60)
    [b for b in at.button if b.label == "😇 Ideal"][0].click().run()
    sin_error(at)
    assert at.session_state["tipo_perturb"] == "Ninguna (solo AWGN)"


@pytest.mark.parametrize("caso", ["9 · Interferencia de tono", "10 · Ruido impulsivo", "11 · Desvanecimiento"])
def test_casos_de_perturbacion(caso):
    at = run()
    [s for s in at.selectbox if s.key == "caso_sel"][0].select(caso).run()
    [b for b in at.button if b.label == "▶ Cargar caso"][0].click().run()
    sin_error(at)
    assert at.session_state["tipo_perturb"] != "Ninguna (solo AWGN)"
