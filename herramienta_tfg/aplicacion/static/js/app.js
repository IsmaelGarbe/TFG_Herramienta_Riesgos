  /* Orquestación de la app */

    document.addEventListener("DOMContentLoaded", () => {
      // Inicializar UI
      UIHelpers.setupDropdowns();
      UIHelpers.bindExclusiveCheckboxes();
      UIHelpers.setupConditionalVisibility();
      UIHelpers.setupNavigation();
      UIHelpers.setupChecklists();
      cargarSelectorEmpresas();
      //Cargar datos guardados
      const all = loadAll();
      const nombres = Object.keys(all);
      //cargar la primera empresa
      const prev = nombres.length ? all[nombres[0]] : {};
      //actualizar el selector
      if (nombres.length) {
        const primera = nombres[0];
        document.getElementById("empresaSelector").value = primera;
        document.querySelector('[name="nombre_empresa"]').value = primera;
      }

      if (prev.corporativos) {
        window.evaluacion.corporativos = prev.corporativos;
        hydrateForm('form-corporativos', prev.corporativos);
      }

      if (prev.infraestructura) {
        window.evaluacion.infraestructura = prev.infraestructura;
        hydrateForm('form-infraestructura', prev.infraestructura);
      }

      if (prev.madurez) {
        window.evaluacion.madurez = prev.madurez;
        hydrateForm('form-madurez', prev.madurez);
      }
    });

  // Hidrata un <form> con valores
  function hydrateForm(formId, values){
    if (!values) return;

    const form = document.getElementById(formId);

    //Rellenar inputs normales
    Object.entries(values).forEach(([name, value])=>{
      const els = form.querySelectorAll(`[name="${CSS.escape(name)}"]`);
      if (!els.length) return;

      const vals = Array.isArray(value) ? value.map(String) : [String(value)];

      els.forEach(el=>{
        const tag = el.tagName.toLowerCase();
        const type= (el.type||'').toLowerCase();

        if (type==='checkbox' || type==='radio') {
          el.checked = vals.includes(el.value);

        } else if (tag==='select' && el.multiple){
          Array.from(el.options).forEach(opt=> opt.selected = vals.includes(opt.value));

        } else {
          el.value = vals[0] ?? '';
        }
      });
    });

    //sincronizar dropdown-checklists (checkboxes visibles)
    const checklistMap = {
      "opera_otrospaises": ".pais-option",
      "activos_criticos": ".activo-option",
      "incidente_sufrido": ".incidente-option",
      "sistemas_operativos": ".sistema-option"
    };

    Object.entries(checklistMap).forEach(([name, selector])=>{
      const val = values[name];
      if (!val) return;

      const vals = Array.isArray(val) ? val.map(String) : String(val).split(",");

      form.querySelectorAll(selector).forEach(el=>{
        el.checked = vals.includes(el.value);
      });
    });

    //refrescar UI dinámica (muy importante)
    form.dispatchEvent(new Event('change', {bubbles:true}));
  }

  const STORAGE_KEY = 'evaluacion_ciberriesgo_v1';
  window.evaluacion = { corporativos:{}, infraestructura:{}, madurez:{} };

  // Carga/guarda localStorage
  function loadState(){
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; }
    catch { return {}; }
  }
  function saveState(nombreEmpresa){
    const all = loadAll();

    all[nombreEmpresa] = window.evaluacion;

    localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  }
  function loadAll(){
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    } catch {
      return {};
    }
  }
  function getNombreEmpresa(){
    const input = document.querySelector('[name="nombre_empresa"]');
    return input?.value || "empresa_sin_nombre";
  }
  function cargarSelectorEmpresas(){
    const select = document.getElementById("empresaSelector");
    if (!select) return;
    const all = loadAll();

    select.innerHTML = '<option value="">Nueva empresa</option>';

    Object.keys(all).forEach(nombre=>{
      const opt = document.createElement("option");
      opt.value = nombre;
      opt.textContent = nombre;
      select.appendChild(opt);
    });

    select.onchange = () => {
      const empresa = select.value;
      if (!empresa) return;

      const allData = loadAll(); // 🔥 siempre datos actualizados
      const data = allData[empresa];

      // Cargar datos en memoria
      window.evaluacion = data;

      // Actualizar input de nombre
      document.querySelector('[name="nombre_empresa"]').value = empresa;

      // Hidratar formularios
      hydrateForm('form-corporativos', data.corporativos);
      hydrateForm('form-infraestructura', data.infraestructura);
      hydrateForm('form-madurez', data.madurez);
    };
  }
  function getFormData(formId){
    const form = document.getElementById(formId);
    const data = new FormData(form);
    const obj = Object.fromEntries(data.entries());

    //dropdowns
    const checklists = [
      { name: "opera_otrospaises", selector: ".pais-option" },
      { name: "activos_criticos", selector: ".activo-option" },
      { name: "incidente_sufrido", selector: ".incidente-option" },
      { name: "sistemas_operativos", selector: ".sistema-option" }
    ];

    checklists.forEach(({name, selector})=>{
      const checked = Array.from(form.querySelectorAll(selector + ":checked"))
        .map(el => el.value);

      if (checked.length) {
        obj[name] = checked; // 👈 array real
      }
    });

    return obj;
  }
  function formatearMoneda(valor){
    if (valor == null) return "-";

    return new Intl.NumberFormat('es-ES', {
      style: 'currency',
      currency: 'EUR',
      maximumFractionDigits: 0
    }).format(valor);
  }

  // ====== Enchufes sobre tus botones ======
  // OJO: este bloque asume que ya existen los elementos en el DOM
  // Lógica del asistente de pasos + validaciones
    const show = el => el.style.display = '';
    const hide = el => el.style.display = 'none';
    const qs  = (s, r=document) => r.querySelector(s);
    const qsa = (s, r=document) => Array.from(r.querySelectorAll(s));

    const steps = [
      {form: qs('#form-corporativos')},
      {form: qs('#form-infraestructura')},
      {form: qs('#form-madurez')}
    ];

    const breadcrumb = qs('#wizardBreadcrumb');
    const bcItems = qsa('.breadcrumb-item', breadcrumb);
    let currentStep = 0;

    function goToStep(idx){
      steps.forEach((s,i)=> i===idx ? show(s.form) : hide(s.form));
      currentStep = idx;

      bcItems.forEach((li,i)=>{
        const link = qs('a', li);
        if (i <= currentStep) {
          li.classList.remove('disabled');
          link.removeAttribute('tabindex');
          link.removeAttribute('aria-disabled');
          link.classList.toggle('fw-semibold', i===currentStep);
        } else {
          li.classList.add('disabled');
          link.setAttribute('tabindex','-1');
          link.setAttribute('aria-disabled','true');
          link.classList.remove('fw-semibold');
        }
        if (i === currentStep) li.classList.add('active'); else li.classList.remove('active');
      });
      window.scrollTo({top:0, behavior:'smooth'});
    }

    // Validar un formulario (incluye checklists/condicionales)
    // Condicionales del módulo 1

    function validateStep(idx){
      const form = steps[idx].form;

      if (idx===0){
        const mat = qs('#materializacion_incidentes').value;
        if (mat==='Sí'){
          qs('#ataques_exitosos_totales').setAttribute('required','required');
          qs('#campoAtaquesExitosos').classList.remove('oculto');
        } else {
          qs('#ataques_exitosos_totales').removeAttribute('required');
          qs('#campoAtaquesExitosos').classList.add('oculto');
        }

        const inc5 = parseInt(qs('#incidentes_ultimos5').value||'0',10);
        if (inc5>0){
          qs('#exitosos_ultimos5').setAttribute('required','required');
          qs('#campoExitososUltimos5').classList.remove('oculto');
        } else {
          qs('#exitosos_ultimos5').removeAttribute('required');
          qs('#campoExitososUltimos5').classList.add('oculto');
        }

        const fact = parseFloat(qs('#facturacion_anual').value||'0');
        if (fact>0){
          qs('#camposFacturacionExtras').classList.remove('oculto');
          ['#pico_maximo','#pico_minimo','#precio_medio_operacion'].forEach(sel=>qs(sel).setAttribute('required','required'));
        } else {
          qs('#camposFacturacionExtras').classList.add('oculto');
          ['#pico_maximo','#pico_minimo','#precio_medio_operacion'].forEach(sel=>qs(sel).removeAttribute('required'));
        }
      }

      // Condicionales del módulo 3
      if (idx===2){
        const soc = qs('#soc').value;
        const boxSoc = qs('#camposSocExtra');
        if (soc==='Externo contratado'){ boxSoc.classList.remove('oculto'); qs('#soc_promedio').setAttribute('required','required'); qs('#soc_maximo').setAttribute('required','required'); }
        else { boxSoc.classList.add('oculto'); qs('#soc_promedio').removeAttribute('required'); qs('#soc_maximo').removeAttribute('required'); }

        const com = qs('#comunicacion').value;
        const boxCom = qs('#camposComunicacionExtra');
        if (com==='Externo contratado'){ boxCom.classList.remove('oculto'); qs('#comunicacion_promedio').setAttribute('required','required'); qs('#comunicacion_maximo').setAttribute('required','required'); }
        else { boxCom.classList.add('oculto'); qs('#comunicacion_promedio').removeAttribute('required'); qs('#comunicacion_maximo').removeAttribute('required'); }

        const as = qs('#asesoria').value;
        const boxAs = qs('#camposAsesoriaExtra');
        if (as==='Externo contratado'){ boxAs.classList.remove('oculto'); qs('#asesoria_promedio').setAttribute('required','required'); qs('#asesoria_maximo').setAttribute('required','required'); }
        else { boxAs.classList.add('oculto'); qs('#asesoria_promedio').removeAttribute('required'); qs('#asesoria_maximo').removeAttribute('required'); }
      }

      const okNative = form.checkValidity();
      form.classList.add('was-validated');

      // Comprobación final de los hidden requeridos
      let hiddenOK = true;
      qsa('input[type="hidden"][required]', form).forEach(h=>{
        if (!h.value || !h.value.trim()) hiddenOK = false;
      });

      return okNative && hiddenOK;
    }

    // Click en breadcrumb
    qsa('.step-link').forEach(link=>{
      link.addEventListener('click', (e)=>{
        e.preventDefault();
        const target = parseInt(link.dataset.step,10);
        if (target === currentStep) return;
        if (target < currentStep) {
          goToStep(target);
        } else {
          let canAdvance = true;
          for (let i=currentStep; i<target; i++){
            if (!validateStep(i)) { canAdvance = false; break; }
          }
          if (canAdvance) goToStep(target);
        }
      });
    });

    // Botones Siguiente / Anterior
    qs('#btnNext0').addEventListener('click', ()=>{
      if (validateStep(0)) {
        window.evaluacion.corporativos = getFormData('form-corporativos');
        saveState(getNombreEmpresa());
        cargarSelectorEmpresas();
        const nombre = getNombreEmpresa();
        document.getElementById("empresaSelector").value = nombre;
        goToStep(1);
      }
    });
    qs('#btnPrev1').addEventListener('click', ()=> goToStep(0));
    qs('#btnNext1').addEventListener('click', ()=>{
      if (validateStep(1)) {
        window.evaluacion.infraestructura = getFormData('form-infraestructura');
        saveState(getNombreEmpresa());
        cargarSelectorEmpresas();
        const nombre = getNombreEmpresa();
        document.getElementById("empresaSelector").value = nombre;
        goToStep(2);
      }
    });
    qs('#btnPrev2').addEventListener('click', ()=> goToStep(1));

    // Finalizar (habilita cálculo)
    qs('#btnFinish').addEventListener('click', ()=>{
      if (validateStep(2)) {
        window.evaluacion.madurez = getFormData('form-madurez');
        saveState(getNombreEmpresa());
        cargarSelectorEmpresas();
        const nombre = getNombreEmpresa();
        document.getElementById("empresaSelector").value = nombre;
        qs('#btnFinish').disabled = true;
        qs('#generarAmenazasBtn').disabled = false;
        alert('Formulario completado. Ya puedes calcular el riesgo.');
      }
    });
    qsa('#form-madurez input, #form-madurez select').forEach(el=>{
      el.addEventListener('input', ()=>{ qs('#btnFinish').disabled = !qs('#form-madurez').checkValidity(); });
      el.addEventListener('change', ()=>{ qs('#btnFinish').disabled = !qs('#form-madurez').checkValidity(); });
    });

    // ====== Dropdown-checklists con feedback en tiempo real ======
    function setupChecklist(rootId, optionClass, hiddenId, textId, fbSelector){
      const root   = qs('#'+rootId);
      const btn    = qs('.dropdown-btn', root);
      const box    = qs('.dropdown-content', root);
      const opts   = qsa('.'+optionClass, root);
      const hidden = qs('#'+hiddenId);
      const text   = qs('#'+textId);
      const fbEl   = fbSelector ? qs(fbSelector) : null;

      // Asegura placeholder base
      if (!text.dataset.placeholder) {
        text.dataset.placeholder = text.textContent.trim() || 'Seleccione al menos una opción';
      }

      btn.addEventListener('click', ()=> root.classList.toggle('open'));

      function refresh(){
        const selected = opts.filter(cb => cb.checked).map(cb => cb.value);
        hidden.value = selected.join(',');
        text.textContent = selected.length ? selected.join(', ') : text.dataset.placeholder;

        // Mostrar/ocultar feedback inmediatamente
        if (fbEl) {
          fbEl.style.display = selected.length ? 'none' : 'block';
        }
      }

      // Exclusivo "Ninguno/No"
      opts.forEach(cb=>{
        cb.addEventListener('change', ()=>{
          if (cb.dataset && cb.dataset.exclusive){
            if (cb.checked){
              opts.forEach(o=>{
                if (o!==cb){ o.checked=false; o.disabled=true; }
              });
            } else {
              opts.forEach(o=> o.disabled=false);
            }
          }
          refresh();
        });
      });

      // Cerrar al hacer click fuera
      document.addEventListener('click', (e)=>{
        if (!root.contains(e.target)) root.classList.remove('open');
      });

      // Inicial
      refresh();
    }

    // Instancias (nota: todos con el mismo mensaje estándar)
    setupChecklist('paisesDropdown',       'pais-option',       'opera_otrospaises',          'paisesDropdownText',       '#opera_otrospaises_fb');
    setupChecklist('activosDropdown',      'activo-option',     'activos_criticos',           'activosDropdownText',      '#activos_fb');
    setupChecklist('incidentesDropdown',   'incidente-option',  'incidente_sufrido',          'incidentesDropdownText',   '#incidentes_fb');
    setupChecklist('sistemasDropdown',     'sistema-option',    'sistemas_operativos',        'sistemasDropdownText',     '#sistemas_fb');

    // ====== Condicionales en tiempo real (paso 1) ======
    qs('#materializacion_incidentes').addEventListener('change', e=>{
      const showExitosos = e.target.value === 'Sí';
      const campo = qs('#campoAtaquesExitosos');
      showExitosos ? campo.classList.remove('oculto') : campo.classList.add('oculto');
    });
    qs('#incidentes_ultimos5').addEventListener('input', e=>{
      const val = parseInt(e.target.value||'0',10);
      const campo = qs('#campoExitososUltimos5');
      val>0 ? campo.classList.remove('oculto') : campo.classList.add('oculto');
    });
    qs('#facturacion_anual').addEventListener('input', e=>{
      const val = parseFloat(e.target.value||'0');
      const campos = qs('#camposFacturacionExtras');
      val>0 ? campos.classList.remove('oculto') : campos.classList.add('oculto');
    });

    // ====== Condicionales (paso 3) ======
    qs('#soc').addEventListener('change', e=>{
      const extra = qs('#camposSocExtra');
      e.target.value==='Externo contratado' ? extra.classList.remove('oculto') : extra.classList.add('oculto');
    });
    qs('#comunicacion').addEventListener('change', e=>{
      const extra = qs('#camposComunicacionExtra');
      e.target.value==='Externo contratado' ? extra.classList.remove('oculto') : extra.classList.add('oculto');
    });
    qs('#asesoria').addEventListener('change', e=>{
      const extra = qs('#camposAsesoriaExtra');
      e.target.value==='Externo contratado' ? extra.classList.remove('oculto') : extra.classList.add('oculto');
    });

    // Estado inicial
    goToStep(0);

    // Botón “Calcular riesgo”

    qs('#generarAmenazasBtn').addEventListener('click', async () => {
      try {
        const data = await FAIR.enviarEvaluacionFAIR();
        console.log("DATA FAIR:", data);
        if (!data.ok) {
          alert("Error en evaluación FAIR");
          return;
        }

        // Mostrar predicción ML
        qs('#resultadoML').style.display = 'block';
        qs('#prediccionTexto').textContent = data.prediccion || "—";

        // Pintar curvas FAIR
        if (data.curvas) {
          Charts.pintarCurvas(data.curvas);
        }

        // (Opcional) resumen textual
        // 🔥 Mostrar métricas FAIR
        if (data.resumen) {
          const r = data.resumen;

          document.getElementById("resultadosRiesgo").style.display = "block";

          document.getElementById("aalValor").textContent = formatearMoneda(r.AAL);
          document.getElementById("p50Valor").textContent = formatearMoneda(r.P50);
          document.getElementById("p95Valor").textContent = formatearMoneda(r.P95);
          document.getElementById("p99Valor").textContent = formatearMoneda(r.P99);
        }

      } catch (err) {
        console.error("Error llamando a FAIR:", err);
        alert("No se pudo calcular el riesgo.");
      }
    });

