#fair backend
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

LEF_MAP = {
    "Menor al 5% (MUY BAJA)": (0.01, 0.025, 0.049),
    "Entre el 5% y el 29% (BAJA)": (0.05, 0.17, 0.29),
    "Entre el 30% y el 50% (MEDIA)": (0.30, 0.40, 0.50),
    "Entre el 51% y el 89% (ALTA)": (0.51, 0.70, 0.89),
    "Entre el 90% y el 95% (MUY ALTA)": (0.90, 0.925, 0.95),
    "Mayor al 95% (CRÍTICA)": (0.96, 0.975, 0.99),
}

@dataclass
class Triplet:
    min: float
    ml: float
    max: float

def _to_num(x) -> float:
    try:
        if x is None: return 0.0
        return float(x)
    except Exception:
        return 0.0

def mapear_lef(lef_respuesta: str) -> tuple[float,float,float]:
    return LEF_MAP.get((lef_respuesta or "").strip(), (0.25, 0.50, 0.75))

def sample_pert(rng: np.random.Generator, a: float, m: float, b: float, lamb: float = 4.0, size=None):
    """
    PERT usando Beta parametrizada (igual que en el JS).
    """
    a, m, b = float(a), float(m), float(b)
    if b <= a:
        return np.full(size or (), max(a,0.0), dtype=float)
    if m < a:
        m=a
    elif m > b:
        m=b
    alpha = 1.0 + lamb * ((m - a) / (b - a))
    beta  = 1.0 + lamb * ((b - m) / (b - a))
    alpha=max(alpha,1e-6)
    beta=max(beta,1e-6)
    # numpy tiene beta directo
    x = rng.beta(alpha, beta, size=size)
    return a + x * (b - a)

def construir_tripletes_desde_formulario(data: dict) -> tuple[tuple[float,float,float], Triplet, tuple[float,float,float], Triplet, float]:
    #LEF (probabilidad impacto primario)
    lef = mapear_lef(data.get("probabilidad_ataque"))

    #LOSS (impacto primario: pérdida negocio y forense)

    # Coste Interrupción del negocio (detección+recuperación)*operaciones*precio_operación
    facturacion = _to_num(data.get("facturacion_anual"))
    pico_max = _to_num(data.get("pico_maximo")) if facturacion > 0 else 0.0
    pico_min = _to_num(data.get("pico_minimo")) if facturacion > 0 else 0.0
    precio_op = _to_num(data.get("precio_medio_operacion")) if facturacion > 0 else 0.0

    rto_min = _to_num(data.get("rto_minimo"))
    rto_ml  = _to_num(data.get("rto_promedio"))
    rto_max = _to_num(data.get("rto_maximo"))

    det_min = _to_num(data.get("deteccion_minima"))
    det_ml  = _to_num(data.get("deteccion_media"))
    det_max = _to_num(data.get("deteccion_maxima"))

    horas_min = (det_min + rto_min) * (pico_min * precio_op)
    horas_ml  = (det_ml  + rto_ml ) * (facturacion / (365.0 * 24.0) if facturacion > 0 else 0.0)
    horas_max = (det_max + rto_max) * (pico_max * precio_op)

    # Coste equipo forense
    soc_tipo = (data.get("soc") or "").strip()
    if soc_tipo == "Externo contratado":
        soc_ml  = _to_num(data.get("soc_promedio"))
        soc_max = max(soc_ml, _to_num(data.get("soc_maximo")))
        forense = Triplet(0.0, soc_ml, soc_max)
    else:
        forense = Triplet(0.0, 0.0, 0.0)

    #Triplete costes primarios
    loss_prim = Triplet(horas_min + forense.min, horas_ml + forense.ml, horas_max + forense.max)

    #SLEF (probabilidad impacto secundario)
    slef = (0.7, 0.9, 1.0)

    #SLOSS (impacto secundario: Legal, Comunicación, Concienciación, Inversión en seguridad, pérdida clientela, Reparación
    # Legal
    asesoria_tipo = (data.get("asesoria") or "").strip()
    if asesoria_tipo == "Externo contratado":
        leg_ml  = _to_num(data.get("asesoria_promedio"))
        leg_max = max(leg_ml, _to_num(data.get("asesoria_maximo")))
        abogados = Triplet(0.0, leg_ml, leg_max)
    else:
        abogados = Triplet(0.0, 0.0, 0.0)

    # Comunicación (nota de prensa, marketing, lavado imagen)
    com_tipo = (data.get("comunicacion") or "").strip()
    if com_tipo == "Externo contratado":
        com_ml  = _to_num(data.get("comunicacion_promedio"))
        com_max = max(com_ml, _to_num(data.get("comunicacion_maximo")))
        rrpp = Triplet(0.0, com_ml, com_max)
    else:
        rrpp = Triplet(0.0, 0.0, 0.0)
    #concienciación
    conc = Triplet(0.0, _to_num(data.get("formacion_promedio")), _to_num(data.get("formacion_maximo")))
    #Inversión en seguridad
    seg  = Triplet(0.0, _to_num(data.get("inversion_ciberseguridad")), _to_num(data.get("presupuesto_defensa")))
    #Pérdida clientela
    clientes = _to_num(data.get("clientes"))
    tarifa_media = _to_num(data.get("tarifa_media"))
    tarifa_max = max(tarifa_media, _to_num(data.get("tarifa_maxima")))
    baja_clientes = Triplet(0.0, clientes * tarifa_media, clientes * tarifa_max)
    #Costes por reparación
    rep = Triplet(0.0, _to_num(data.get("presupuesto_reparacion")), _to_num(data.get("activo_mas_costoso")))
    #Tripletes costes secundarios
    sloss = Triplet(
        abogados.min + rrpp.min + conc.min + seg.min + baja_clientes.min + rep.min,
        abogados.ml  + rrpp.ml  + conc.ml  + seg.ml  + baja_clientes.ml  + rep.ml,
        abogados.max + rrpp.max + conc.max + seg.max + baja_clientes.max + rep.max,
    )

    apetito = _to_num(data.get("apetito_riesgo"))
    return lef, loss_prim, slef, sloss, apetito

def simulacion_FAIR(data: dict, iteraciones: int = 10000, seed: int = 42):
    generador_random_num = np.random.default_rng(seed)
    lef, loss_prim, slef, sloss, apetito = construir_tripletes_desde_formulario(data)
    #LEF
    lef_event = sample_pert(generador_random_num, lef[0], lef[1], lef[2], size=iteraciones)
    ocurre = generador_random_num.random(iteraciones) < lef_event
    perdidas = np.zeros(iteraciones, dtype=float)
    # Evaluación del impacto primario
    pLoss = sample_pert(generador_random_num, loss_prim.min, loss_prim.ml, loss_prim.max, size=iteraciones)
    perdidas += np.where(ocurre, np.maximum(0.0, pLoss), 0.0)
    #Secondary LEF
    slef_event = sample_pert(generador_random_num, slef[0], slef[1], slef[2], size=iteraciones)
    ocurre_sec = ocurre & (generador_random_num.random(iteraciones) < slef_event)
    #Evaluación del impacto secundario
    sLoss = sample_pert(generador_random_num, sloss.min, sloss.ml, sloss.max, size=iteraciones)
    perdidas += np.where(ocurre_sec, np.maximum(0.0, sLoss), 0.0)
    perdidas.sort()

    #Cálculo de pérdida media anual esperada
    aal = float(perdidas.mean())
    #Cálculo de percentiles
    def pct(p): return float(np.percentile(perdidas, p))
    # Curvas
    max_loss = float(perdidas.max()) if iteraciones else 0.0
    # Loss Magnitude/yr:
    bins_lm = 25
    edges = np.linspace(0.0, max_loss if max_loss > 0 else 1.0, bins_lm + 1)
    hist, _ = np.histogram(perdidas, bins=edges)
    prob_lm = (hist / hist.sum() * 100.0).tolist()
    labels_lm = [float(round(edges[i], 0)) for i in range(bins_lm)]  # inicio de bin
    # Chance of Exceedance:
    bins_ex = 50
    thresholds = np.linspace(apetito if apetito > 0 else max_loss, 0.0, bins_ex + 1)
    # uso de searchsorted sobre array ordenado
    idxs = np.searchsorted(perdidas, thresholds, side="left")
    prob_ex = ((len(perdidas) - idxs) / len(perdidas) * 100.0).tolist()
    labels_ex = thresholds.tolist()

    p_excede_apetito = 0.0
    if apetito > 0:
        i = np.searchsorted(perdidas, apetito, side="left")
        p_excede_apetito = float((len(perdidas) - i) / len(perdidas) * 100.0)

    return {
        "perdidas": perdidas,  #depurar
        "curvas": {
            "lossMagnitude": {"labels": labels_lm, "probs": prob_lm},
            "exceedance": {"labels": labels_ex, "probs": prob_ex, "pExcedeApetito": p_excede_apetito},
        },
        "resumen": {"AAL": aal, "P50": pct(50), "P95": pct(95), "P99": pct(99)},
    }
