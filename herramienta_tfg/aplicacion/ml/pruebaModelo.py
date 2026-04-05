from pathlib import Path

from PIL.Image import ENCODERS
from joblib import load
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from aplicacion.modelo_ml import tratarValoresExtremos, normalizarDatos #, codificarVariablesCategoricas

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "casosSinteticosRiesgos.csv"
#ENCODERS_PATH = BASE_DIR / "ml" / "encoders.joblib"
MODEL_PATH = "modelo_entrenado.joblib"
PRUEBA_PATH= "pruebaTest.csv"
model = load(MODEL_PATH)
MODEL_FEATURES_PATH = BASE_DIR / "model_features.joblib"
model_features = load(MODEL_FEATURES_PATH)

# 2. Leer CSV y tomar la última fila
df = pd.read_csv(PRUEBA_PATH, delimiter=";", encoding="cp1252")
#cargar diccionario encoders persistente
#encoders = load(ENCODERS_PATH)
ultima_fila = df.index[-1]
nueva_fila = df.tail(1).drop(columns=["evento_amenaza"], errors="ignore")
df_train = pd.read_csv(CSV_PATH, delimiter=";", encoding="cp1252")

for col in nueva_fila.columns:
    if nueva_fila[col].dtype == object:
        freq = df_train[col].value_counts(normalize=True)
        raras = freq[freq < 0.02].index
        nueva_fila[col] = nueva_fila[col].replace(raras, 'OTRA')

"""def aplicarCodificacionPersistente(df, encoders):
    # Recorre el dataframe y aplica la codificación guardada
    for col in df.columns:
        if col in encoders:
            le = encoders[col]
            df[col] = df[col].astype(str)

            # GESTIÓN DE CATEGORÍAS NO VISTAS
            new_labels = df[col].loc[~df[col].isin(le.classes_)]
            if not new_labels.empty:
                # Para categorías nuevas, asignamos el código de la clase más frecuente
                df.loc[new_labels.index, col] = le.classes_[0]

            df[col] = le.transform(df[col])

    return df"""
# 3. Preprocesado igual que en entrenamiento
#nueva_fila = codificarVariablesCategoricas(nueva_fila)
#nueva_fila= aplicarCodificacionPersistente(nueva_fila, encoders)
nueva_fila = pd.get_dummies(nueva_fila, drop_first=False)
nueva_fila = nueva_fila.astype(int)
nueva_fila = tratarValoresExtremos(nueva_fila)
faltantes = list(set(model_features) - set(nueva_fila.columns))

if faltantes:
    nueva_fila = pd.concat([nueva_fila, pd.DataFrame(0, index=nueva_fila.index, columns=faltantes)],axis=1)

nueva_fila = nueva_fila[model_features]
#nueva_fila = normalizarDatos(nueva_fila)
# 4. Ajustar columnas a las que el modelo espera
#model_features = model.get_booster().feature_names

#El problema esta aqui (fuerza a cero todas las columnas que el modelo espera, pero que no están presentes tras el preprocesado de la nueva fila. Es decir, si la fila de entrada no contiene exactamente los mismos valores codificados (mismo número de categorías por columna) que durante el entrenamiento, faltarán columnas que se rellenan a cero.
#Esto convierte toda la fila de entrada en una fila muy "neutral", muy genérica, parecida a otras entradas sintéticas repetidas en el dataset, y el modelo, ante esa entrada sin señales claras, predice sistemáticamente la clase más común tras SMOTE, que es justo Fraude (clase 3), ya que se ha igualado su frecuencia con otras clases.

#nueva_fila = nueva_fila.reindex(columns=model_features, fill_value=0)

#solucion, reordenamiento seguro, que detecte si hay columnas que no existen y las gestione adecuadamente antes del reindex

print("Comprobacion")
print("Valores únicos:", nueva_fila.nunique(axis=1).values)
print("Suma fila:", nueva_fila.sum(axis=1).values)



# 5. Hacer predicción
#pred = model.predict(nueva_fila)[0]
print("Comprobacion valores")
print(nueva_fila.head(1).T)
"""evento_amenaza= {
    0: "Ransomware",
    2: "Brecha de datos",
    3: "Fraude",
    4: "Destrucción física"
}
try:
    codigo = int(float(pred))
    etiqueta = evento_amenaza.get(codigo, f"Desconocido ({codigo})")
except Exception:
    etiqueta = "Desconocido"
"""
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder_y.joblib"
label_encoder_y = load(LABEL_ENCODER_PATH)

pred = model.predict(nueva_fila)[0]
etiqueta = label_encoder_y.inverse_transform([pred])[0]
probas = model.predict_proba(nueva_fila)[0]

for clase, p in zip(label_encoder_y.classes_, probas):
    print(f"{clase}: {p:.3f}")
# 8. Mostrar resultado en consola (opcional)
print(etiqueta)


