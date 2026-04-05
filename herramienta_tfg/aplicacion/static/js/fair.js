/* =========================================================
   FAIR.JS (FRONTEND)
   ---------------------------------------------------------
   El cálculo FAIR (LEF, LOSS, SLEF, SLOSS y Monte Carlo)
   se realiza EXCLUSIVAMENTE en el backend (fair.py).

   Este archivo:
   - recoge datos del formulario
   - construye el payload
   - envía la evaluación al backend
   - NO realiza ningún cálculo de riesgo
   ========================================================= */

/* =========================================================
   1 Obtener datos del formulario FAIR
   ========================================================= */
function obtenerDatosFormularioFAIR() {
  const get = (id) => document.getElementById(id)?.value ?? null;

  return {
    // --- Probabilidad / LEF ---
    probabilidad_ataque: get("probabilidad_ataque"),

    // --- Negocio ---
    facturacion_anual: get("facturacion_anual"),
    pico_minimo: get("pico_minimo"),
    pico_maximo: get("pico_maximo"),
    precio_medio_hora: get("precio_medio_operacion"),

    // --- Detección y recuperación ---
    deteccion_minima: get("deteccion_minima"),
    deteccion_media: get("deteccion_media"),
    deteccion_maxima: get("deteccion_maxima"),

    rto_minimo: get("rto_minimo"),
    rto_promedio: get("rto_promedio"),
    rto_maximo: get("rto_maximo"),

    // --- SOC / Forense ---
    soc: get("soc"),
    soc_promedio: get("soc_promedio"),
    soc_maximo: get("soc_maximo"),

    // --- Impactos secundarios ---
    asesoria: get("asesoria"),
    asesoria_promedio: get("asesoria_promedio"),
    asesoria_maximo: get("asesoria_maximo"),

    comunicacion: get("comunicacion"),
    comunicacion_promedio: get("comunicacion_promedio"),
    comunicacion_maximo: get("comunicacion_maximo"),

    formacion_promedio: get("formacion_promedio"),
    formacion_maximo: get("formacion_maximo"),

    inversion_ciberseguridad: get("inversion_ciberseguridad"),
    presupuesto_defensa: get("presupuesto_defensa"),

    clientes: get("clientes"),
    tarifa_media: get("tarifa_media"),
    tarifa_maxima: get("tarifa_maxima"),

    presupuesto_reparacion: get("presupuesto_reparacion"),
    activo_mas_costoso: get("activo_mas_costoso"),

    // --- Apetito de riesgo ---
    apetito_riesgo: get("apetito_riesgo")
  };
}


/* =========================================================
   2 Enviar evaluación FAIR al backend
   =========================================================*/
async function enviarEvaluacionFAIR(endpoint = "/api/finalizar-evaluacion") {
  const datos = obtenerDatosFormularioFAIR();
  const form1 = Object.fromEntries(new FormData(qs('#form-corporativos')).entries());
  const form2 = Object.fromEntries(new FormData(qs('#form-infraestructura')).entries());
  const form3 = Object.fromEntries(new FormData(qs('#form-madurez')).entries());

  const payload = { ...form1, ...form2, ...form3, ...datos };

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error("Error al enviar la evaluación FAIR al backend");
  }

  return await response.json();
}

/* =========================================================
   3 API mínima expuesta al frontend
   ========================================================= */
window.FAIR = {
  enviarEvaluacionFAIR
};
