# aplicacion/ml/prediccion.py
from pathlib import Path
from joblib import load
import pandas as pd
import numpy as np
from aplicacion.modelo_ml import tratarValoresExtremos

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "modelo_entrenado.joblib"
MODEL_FEATURES_PATH = BASE_DIR / "model_features.joblib"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder_y.joblib"
CSV_PATH = BASE_DIR / "casosSinteticosRiesgos.csv"

def predecir():

    model = load(MODEL_PATH)
    model_features = load(MODEL_FEATURES_PATH)
    label_encoder_y = load(LABEL_ENCODER_PATH)

    #Leer CSV
    df = pd.read_csv(CSV_PATH, delimiter=";", encoding="cp1252")

    if df.empty:
        raise ValueError("El CSV está vacío, no hay datos para predecir")

    nueva_fila = df.tail(1).drop(columns=["evento_amenaza"], errors="ignore")

    #One-hot encoding
    nueva_fila = pd.get_dummies(nueva_fila, drop_first=False)
    nueva_fila = nueva_fila.astype(int)

    #Valores extremos
    nueva_fila = tratarValoresExtremos(nueva_fila)

    #Alinear columnas con el modelo
    faltantes = set(model_features) - set(nueva_fila.columns)
    for col in faltantes:
        nueva_fila[col] = 0

    nueva_fila = nueva_fila[model_features]
    #Predicción
    pred_code = model.predict(nueva_fila)[0]
    etiqueta = label_encoder_y.inverse_transform([pred_code])[0]

    return etiqueta
