from flask import Flask, render_template, request, jsonify
from aplicacion.ml.prediccion import predecir
from aplicacion.fair import simulacion_FAIR
import csv
import os
import unicodedata
from pathlib import Path
import logging

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# ---------------- FLASK ----------------
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.logger.setLevel(logging.DEBUG)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent
ML_DIR = BASE_DIR / "ml"
CSV_PATH = ML_DIR / "casosSinteticosRiesgos.csv"

CSV_DELIMITER = ";"
CSV_ENCODING = "cp1252"

# ---------------- CSV COLUMNS ----------------
CSV_COLS = [
    "nombre_empresa","pais","opera_otrospaises","sector_empresa","numero_empleados",
    "infraestructura_critica","activos_criticos","pilares_prioritarios","actor_amenaza",
    "incidente_sufrido","pilar_critico","sistemas_operativos","despliegue_sistemas",
    "centros_datos","segmentacion_red","vpn","firewall","monitorizacion_red",
    "balanceadores","antiphishing","edr","actualizaciones","frecuencia_actualizaciones",
    "autenticacion","politica_contraseñas","iam","revocacion","backups","evento_amenaza"
]

# ---------------- HELPERS ----------------
def nfc(s):
    if s is None:
        return ''
    return unicodedata.normalize('NFC', str(s))

def ensure_csv_header():
    if not CSV_PATH.exists():
        ML_DIR.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="", encoding=CSV_ENCODING) as f:
            writer = csv.writer(f, delimiter=CSV_DELIMITER)
            writer.writerow(CSV_COLS)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    ensure_csv_header()
    return render_template("index.html")

@app.route("/api/finalizar-evaluacion", methods=["POST"])
def finalizar_evaluacion():
    try:
        data = request.get_json(force=True) or {}

        # 1 Guardar fila con la evaluación en CSV
        row = [nfc(str(data.get(col, "")).strip()) for col in CSV_COLS]

        with CSV_PATH.open("a", newline="", encoding=CSV_ENCODING) as f:
            writer = csv.writer(f, delimiter=CSV_DELIMITER)
            writer.writerow(row)

        # 2️ Llamada a la función de predicción del modelo entrenado
        amenaza = predecir()

        # 3️ Llamada la función FAIR
        fair_out = simulacion_FAIR(data, 10000)
        curvas = fair_out["curvas"]
        resumen=fair_out["resumen"]

        #Devuelve el resultado
        return jsonify({
            "ok": True,
            "prediccion": amenaza,
            "curvas": curvas,
            "resumen": resumen
        })

    except Exception as e:
        app.logger.exception("Error en finalizar_evaluacion")
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/ping")
def ping():
    return jsonify({"pong": True})

# ---------------- MAIN ----------------
if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
