/* Utilidades de UI y formularios (desplegables, visibilidad condicional, etc.) */

function toggleDropdownById(id) {
  const el = document.getElementById(id);
  if (el) el.classList.toggle("open");
}

function setupDropdowns() {
  document.querySelectorAll(".dropdown-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-dropdown");
      if (id) toggleDropdownById(id);
    });
  });
}

function bindExclusiveCheckboxes() {
  // Checkbox con data-exclusive="#contenedor" desactiva el resto al marcarse.
  document.querySelectorAll("input[type='checkbox'][data-exclusive]").forEach(exclusiveCb => {
    exclusiveCb.addEventListener("change", () => {
      const containerSel = exclusiveCb.getAttribute("data-exclusive");
      const container = document.querySelector(containerSel);
      if (!container) return;
      const cbs = container.querySelectorAll("input[type='checkbox']");
      if (exclusiveCb.checked) {
        cbs.forEach(cb => { if (cb !== exclusiveCb) { cb.checked = false; cb.disabled = true; } });
      } else {
        cbs.forEach(cb => { if (cb !== exclusiveCb) cb.disabled = false; });
      }
    });
  });
}

function updateHiddenFromChecklist(containerId, checkboxSelector, hiddenInputId, labelSpanId, emptyText) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const checkboxes = container.querySelectorAll(checkboxSelector);
  const hidden = document.getElementById(hiddenInputId);
  const label = document.getElementById(labelSpanId);

  const sync = () => {
    const seleccionados = Array.from(checkboxes).filter(cb => cb.checked).map(cb => cb.value);
    if (hidden) hidden.value = seleccionados.join(",");
    if (label) label.textContent = seleccionados.length ? seleccionados.join(", ") : emptyText;
  };

  checkboxes.forEach(cb => cb.addEventListener("change", sync));
  sync();
}

/* Visibilidad condicional de campos */
function setupConditionalVisibility() {
  // 13. materialización -> campoAtaquesExitosos
  const materializacion = document.getElementById("materializacion_incidentes");
  const campoAtaquesExitosos = document.getElementById("campoAtaquesExitosos");
  const ataquesExitososTotales = document.getElementById("ataques_exitosos_totales");
  if (materializacion && campoAtaquesExitosos) {
    materializacion.addEventListener("change", () => {
      const mostrar = materializacion.value === "Sí";
      campoAtaquesExitosos.classList.toggle("oculto", !mostrar);
      if (!mostrar && ataquesExitososTotales) ataquesExitososTotales.value = "";
    });
  }

  // 14. incidentes últimos 5 años -> campoExitososUltimos5
  const incidentes5 = document.getElementById("incidentes_ultimos5");
  const campoExitosos5 = document.getElementById("campoExitososUltimos5");
  const exitosos5 = document.getElementById("exitosos_ultimos5");
  if (incidentes5 && campoExitosos5) {
    incidentes5.addEventListener("change", () => {
      const v = parseInt(incidentes5.value, 10);
      const mostrar = !isNaN(v) && v > 0;
      campoExitosos5.classList.toggle("oculto", !mostrar);
      if (!mostrar && exitosos5) exitosos5.value = "";
    });
  }

  // 18. facturación -> camposFacturacionExtras
  const facturacion = document.getElementById("facturacion_anual");
  const camposFact = document.getElementById("camposFacturacionExtras");
  if (facturacion && camposFact) {
    facturacion.addEventListener("change", () => {
      const val = parseFloat(facturacion.value);
      const mostrar = !isNaN(val) && val > 0;
      camposFact.classList.toggle("oculto", !mostrar);
      if (!mostrar) {
        ["pico_maximo","pico_minimo","precio_medio_hora"].forEach(id => {
          const el = document.getElementById(id); if (el) el.value = "";
        });
      }
    });
  }

  // SOC / Comunicación / Asesoría (mostrar campos extra si Externo contratado)
  const soc = document.getElementById("soc");
  const socExtra = document.getElementById("camposSocExtra");
  if (soc && socExtra) {
    soc.addEventListener("change", () => {
      socExtra.classList.toggle("oculto", soc.value !== "Externo contratado");
      if (soc.value !== "Externo contratado") {
        ["soc_promedio","soc_maximo"].forEach(id => { const el = document.getElementById(id); if (el) el.value = ""; });
      }
    });
  }

  const comunicacion = document.getElementById("comunicacion");
  const comExtra = document.getElementById("camposComunicacionExtra");
  if (comunicacion && comExtra) {
    comunicacion.addEventListener("change", () => {
      comExtra.classList.toggle("oculto", comunicacion.value !== "Externo contratado");
      if (comunicacion.value !== "Externo contratado") {
        ["comunicacion_promedio","comunicacion_maximo"].forEach(id => { const el = document.getElementById(id); if (el) el.value = ""; });
      }
    });
  }

  const asesoria = document.getElementById("asesoria");
  const aseExtra = document.getElementById("camposAsesoriaExtra");
  if (asesoria && aseExtra) {
    asesoria.addEventListener("change", () => {
      aseExtra.classList.toggle("oculto", asesoria.value !== "Externo contratado");
      if (asesoria.value !== "Externo contratado") {
        ["asesoria_promedio","asesoria_maximo"].forEach(id => { const el = document.getElementById(id); if (el) el.value = ""; });
      }
    });
  }
}

/* Navegación entre módulos / formularios */
function setupNavigation() {
  const popup = document.getElementById("popup");
  const btnComenzar = document.getElementById("btnComenzar");
  const seleccionModulo = document.getElementById("seleccion-modulo");
  const accionesEval = document.getElementById("acciones-evaluacion");
  const btnCalcular = document.getElementById("generarAmenazasBtn");

  if (btnComenzar) {
    btnComenzar.addEventListener("click", () => {
      if (popup) popup.style.display = "block";
    });
  }

  document.querySelectorAll(".btnMostrarFormulario").forEach(btn => {
    btn.addEventListener("click", () => {
      const formId = btn.getAttribute("data-form");
      document.querySelectorAll(".formulario").forEach(f => f.style.display = "none");
      const form = document.getElementById(formId);
      if (form) form.style.display = "block";
      if (seleccionModulo) seleccionModulo.classList.remove("active");
      if (accionesEval) accionesEval.style.display = "block";
      if (btnCalcular) btnCalcular.style.display = "inline-block";
    });
  });

  document.querySelectorAll(".btnVolver").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".formulario").forEach(f => f.style.display = "none");
      if (seleccionModulo) seleccionModulo.classList.add("active");
    });
  });

  const enviarEvaluacionBtn = document.getElementById("enviarEvaluacionBtn");
  if (enviarEvaluacionBtn) {
    enviarEvaluacionBtn.addEventListener("click", () => {
      // Por si quieres agregar validaciones globales antes de enviar
      const visibleForm = Array.from(document.querySelectorAll(".formulario"))
        .find(f => f.style.display !== "none");
      if (visibleForm) visibleForm.submit();
    });
  }
}

/* Inicialización de checklists (sin duplicar funciones) */
function setupChecklists() {
  updateHiddenFromChecklist("paisesDropdown", ".pais-option", "opera_otrospaises", "paisesDropdownText", "Seleccione uno o varios países");
  updateHiddenFromChecklist("regulacionesDropdown", ".regulacion-option", "regulaciones_proteccion_datos", "regulacionesDropdownText", "Seleccione una o varias regulaciones");
  updateHiddenFromChecklist("activosDropdown", ".activo-option", "activos_criticos", "activosDropdownText", "Seleccione uno o varios activos");
  updateHiddenFromChecklist("incidentesDropdown", ".incidente-option", "incidente_sufrido", "incidentesDropdownText", "Seleccione uno o varios tipos de incidente");
  updateHiddenFromChecklist("sistemasDropdown", ".sistema-option", "sistemas_operativos", "sistemasDropdownText", "Seleccione uno o varios sistemas operativos");
}

/* Exponer mínima API para otros módulos si hace falta */
window.UIHelpers = {
  setupDropdowns,
  bindExclusiveCheckboxes,
  setupConditionalVisibility,
  setupNavigation,
  setupChecklists
};
