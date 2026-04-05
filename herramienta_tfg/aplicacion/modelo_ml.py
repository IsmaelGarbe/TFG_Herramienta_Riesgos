# Librerías
from PIL.Image import ENCODERS
from collections import Counter

from joblib import load, dump
from pathlib import Path
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from prettytable import PrettyTable
from sklearn.metrics import classification_report #conocer las clases
from imblearn.over_sampling import SMOTE #mayor precision
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Cargar datos
dtypes = {
    "nombre_empresa": str, "pais": str, "opera_otrospaises": str, "sector_empresa": str,
    "numero_empleados": str, "infraestructura_critica": str, "activos_criticos": str,
    "pilares_prioritarios": str, "actor_amenaza": str, "incidente_sufrido": str,
    "pilar_critico": str, "sistemas_operativos": str, "despliegue_sistemas": str,
    "centros_datos": str, "segmentacion_red": str, "vpn": str, "firewall": str,
    "monitorizacion_red": str, "balanceadores": str, "antiphishing": str, "edr": str,
    "actualizaciones": str, "frecuencia_actualizaciones": str, "autenticacion": str,
    "politica_contraseñas": str, "iam": str, "revocacion": str, "backups": str,
    "evento_amenaza": str
}

def cargar_datos(ruta_csv):
    return pd.read_csv(ruta_csv, dtype=dtypes, delimiter=';', encoding='latin1')

# Preparar datos
def eliminarColumnasVacias(df):
    valores_unicos = df.nunique()
    columnas_unicas = df.loc[:, valores_unicos == 1]
    #axis=1 significa que es una columna lo que s elimina (porque valores unicos no aportan informacion al modelo de prediccion)
    return df.drop(columnas_unicas, axis=1)

def missingValues(df):
    return df.isnull().any().any()

def fillMissingValues(df):
    df.fillna(df.mean(), inplace=True)
    return df

"""def dibujarMatrizCorrelación(df):
    # Selecciona solo las columnas numéricas
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    plt.figure(figsize=(40, 40))
    sns.heatmap(corr, annot=True)
    plt.savefig("correlation_matrix.svg", format='svg')
    plt.show()"""

def eliminarColumnasRedundantes(df):
    #eliminar redundancias en columnas altamente correlacionadas (mayor al 0.80 de correlación)
  # Calcular la matriz de correlación
  corr_matrix = df.corr().abs()
  # np.triu matriz triangular de 1s y 0s
  # Seleccionar la diagonal superior de la matriz de correlación ya que es simetrica y no hace falta un analisis duplicado
    #K=1 omite la diagonal, correlacion de una variable consigo misma es siempre 1
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  # Listar columnas con una correlación mayor a 0.80 (redundantes)
  to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
  # Elimina las columnas seleccionadas (las redundantes listadas en to_drop)
  return df.drop(df[to_drop], axis=1)

def codificarVariablesCategoricas(df, encoders=None, training=True):
    #Codifica variables categóricas. Si training=True, crea nuevos LabelEncoders y los guarda. Si training=False, usa los encoders provistos.
    if training:
        encoders={}
    #recorre el dataset y comprueba si los valores son de tipo object (texto o cadenas de caracteres) para convertirlos en valores numericos
    for col in df.columns:
        if df[col].dtype == 'object':
            #df[col] = pd.factorize(df[col])[0]
            df[col]= df[col].astype(str)
            if training:
                le= LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col]=le
            else:
                # USAMOS el diccionario de códigos ya guardado
                le = encoders.get(col)
                if le is not None:
                    # GESTIÓN DE CATEGORÍAS NO VISTAS (Handle unknown)
                    # Añade temporalmente etiquetas no vistas para evitar un error en .transform()
                    new_labels = df[col].loc[~df[col].isin(le.classes_)]
                    if not new_labels.empty:
                    # Rellena las nuevas categorías con el código de la categoría más frecuente (o -1 si no existe)
                        # Esto asegura que transform no falle, aunque la predicción será incierta.
                        df.loc[new_labels.index, col] = le.classes_[0]
                    df[col] = le.transform(df[col])
    # Se devuelven los datos codificados y el diccionario de encoders
    return df, encoders

def hayValoresExtremos(df):
    #identifica si hay valores infinitos, nulos o excesivamente grandes
  mask = np.isinf(df) | np.isnan(df) | (df > 1e5)
  return mask.sum().sum() > 0
def tratarValoresExtremos(df):
    #reemplaza valores infinitos por nulos (NaN)
    df.replace(np.inf, np.nan, inplace=True)
    #reemplaza los negativos infinitos
    df.replace(-np.inf, np.nan, inplace=True)
    #rellena los valores nulos con la media de la columna numerica
    df.fillna(df.mean(), inplace=True)
    #devuelve conjunto de datos sin infinitos
    return df

def normalizarDatos(df, scaler=None, training=True):
    #comprobar variable objetivo si está en el conjunto de datos (para no normalizarla ya que es la que se quiere predecir)
    if 'evento_amenaza' in df.columns:
        #si está guardamos la columna en target
        target = df['evento_amenaza']
        #se elimina la columna temporalmente para que no se vea afectada por la normalización
        df = df.drop('evento_amenaza', axis=1)
    else:
        #si no está la variable objetivo ponemos el valor de target a None (control de errores)
        target = None
    if training:
        # escalador MInMaxScaler que transforma valores de cada columna para estar dentro del rango [0,1]
        scaler = MinMaxScaler()
        # ajuste y transformacion del conjunto de datos pata devolver un array con valores normalizados
        scaled= scaler.fit_transform(df)
    else:
        scaled = scaler.transform(df)
    #se vuelve a convertir el array ya normalizado en un nuevo conjunto de datos asignandole los mismos nombres de las columnas originales
    df_scaled = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    #si se habia separado la columna de la variable objetivo se vuelve a agregar al conjunto de datos ya normalizado
    if target is not None:
        df_scaled['evento_amenaza'] = target
    #devuelve el conjunto de datos con los valores normalizados entre 0 y 1 sin afectar a la columna objetivo
    return df_scaled, scaler

def prepararDatos(df):
    print("Eliminando columnas vacías...")
    df = eliminarColumnasVacias(df)

    print("Rellenando valores vacíos...")
    if missingValues(df):
        df = fillMissingValues(df)

    #print("Codificando variables categóricas...")
    #df, encoders = codificarVariablesCategoricas(df, training=True)
    #y = df['evento_amenaza']
    #df = df.drop('evento_amenaza', axis=1)
    label_encoder_y = LabelEncoder()
    #y = label_encoder_y.fit_transform(df['evento_amenaza'])
    y_encoded = label_encoder_y.fit_transform(df['evento_amenaza'])

    # Convertir a Series para mantener .iloc
    y = pd.Series(y_encoded, index=df.index, name="evento_amenaza")
    # Guardar encoder para predicción
    dump(label_encoder_y, "ml/label_encoder_y.joblib")

    df = df.drop('evento_amenaza', axis=1)
    for col in df.select_dtypes(include='object'):
        freq = df[col].value_counts(normalize=True)
        raras = freq[freq < 0.02].index
        df[col] = df[col].replace(raras, 'OTRA')
    print("Aplicando One-Hot Encoding...")
    df = pd.get_dummies(df, drop_first=False)

    """print("Dibujando matriz de correlación...")
    dibujarMatrizCorrelación(df)"""

    print("Eliminando columnas redundantes...")
    df = eliminarColumnasRedundantes(df)

    print("Tratando valores extremos...")
    df = tratarValoresExtremos(df)

    print("Normalizando datos...")
    #df = normalizarDatos(df)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )

    dump(scaler, "ml/scaler.joblib")
    dump(df_scaled.columns.tolist(), "ml/model_features.joblib")
    #df=df_scaled
    """if 'evento_amenaza' not in df.columns:
        raise ValueError("La columna 'evento_amenaza' no está presente en los datos.")"""

    #separa las columnas (independientes) que el modelo va a analizar en X y la variable objetivo (dependiente) a predecir en Y
    """X = df.drop(['evento_amenaza'], axis=1)
    y = df['evento_amenaza']"""
    #X= df
    print("Número de columnas resultantes: ", df_scaled.shape[1], "\n")

    return df_scaled, y#, encoders

#Algoritmos de prediccion
def modeloXGBoost():
    return XGBClassifier(
        #clasificacion multiclase (probabilidad de cada clase)
        objective='multi:softprob',
        #numero de clases
        num_class=5,
        #aprendizaje mas lento pero con mayor precision evita memorizar (elegido para reducir sobreajuste)
        learning_rate=0.08,
        #profundidad maxima de cada arbol de decision (valor pequeño 2 para evitar sobreajuste) cambio a 4 para evitar subajuste
        max_depth=2,
        #numero de arboles para captar patrones sin añadir demasiada complejidad, cambio de 100 a 120 para mas robustez compensando la reducción en learning_rate y darle más oportunidades de aprendizaje general.
        n_estimators=120,
        #porcentaje de datos usados en cada iteracion del entrenamiento. Al ser 0.8 tiene aleatoriedad lo que reduce el sobreajuste
        subsample=0.8,
        #columnas seleccionadas al azar para entrenar cada arbol
        colsample_bytree=0.8,
        #parametro de regularizacion que introduce penalizacion L1 sobre los coeficientes ayudando a eliminar variables menos relevantes (ruido) y sobreajuste
        reg_alpha=0.0,
        #penaliza grandes coeficientes favorece un modelo mas estable y generalizable (aleatoriedad controlada) cambio de 1.0 a 2.0 para regularización más estricta
        reg_lambda=2.0,
        #Aumentar min_child_weight. Exige más muestras en un nodo terminal antes de realizar una division, ayuda a prevenir el aprendizaje de ruido.
        min_child_weight=10,
        gamma=0.0,
        #semilla aleatoria para resultados reproducibles validacion cruzada
        random_state=42,
        #mide calidad de las probabilidades predichas (mejor cuanto mas bajo) representa el error del modelo al predecir clases
        eval_metric='mlogloss',
        #use_label_encoder=False
    )

def randomForest():
    return RandomForestClassifier(
        n_estimators=400,
        criterion='gini',
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.5,
        class_weight=None, #uso Smote
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

def knn():
  return KNeighborsClassifier(
      #valor bajo para reducir underfitting pero sin memorizar
      n_neighbors=16,
      #estructura eficiente para datos de baja o media prediccion (rapido y eficiente en conjuntos de datos pequeños) encuentra rapidamente distancias de muchas caracteristicas sin tener que comparar con todos los vecinos
      algorithm='kd_tree',
      #numero minimo de puntos en un nodo hoja del arbol KD (valor moderado) mas pequeño es mas preciso pero mas lento
      leaf_size=20,
      #distancia Manhattan (mejor para datos con variables categoricas que la distancia Euclidiana)
      p=1
  )
def svm():
    return SVC(
        #kernel gaussiano (transforma datos no lineales a un espacio que si se puede separar linealmente)
        kernel='rbf',
        #penalizacion por errores de clasificacion equilibrado, valor bajo (evita sobreajuste) valor alto (sobreajuste)
        C=0.5,
        #mide distancia entre puntos (tiene en cuenta la varianza lo que lo hace mas estable y preciso) reduce sobreajuste
        gamma=0.01,
        class_weight='balanced',
        #activa calculo de probabilidades de prediccion pero entrenamiento mas lento (validacion cruzada)
        probability=True,
        #validacion cruzada
        random_state=42
    )

#Entrenamiento
def entrenamiento(model, X_train, y_train):
    #ajusta el modelo aprendiendo patrones de X (datos de entrada) Y (etiquetas)
    #model.fit(X_train, y_train)
    model.fit(X_train, y_train)

    #devuelve el modelo ya entrenado para evaluarse
    return model

def evaluarRendimiento(model, X_test, y_test):
    #evalua el modelo usando el conjunto de test
    y_pred = model.predict(X_test)
    #precision mide cuantos de los eventos predichos pertenecientes a una clase son realmente de esa clase (predicciones reales de esa clase /predicciones que hizo el modelo de esa clase en total)
    #average=weighted calcula media ponderada de cada clase y zero_division=0 evita errores si hay clases que no se predijeron
    precision = precision_score(y_test, y_pred,average='weighted',zero_division=0)
    #mide cuantos de los eventos reales fueron predichos correctamente por el modelo (clases acertadas por el modelo / clases totales reales en el dataset)
    recall = recall_score(y_test, y_pred,average='weighted',zero_division=0)
    #matriz de confusion (predicciones del modelo vs realidad)
    conf_matrix = confusion_matrix(y_test, y_pred)
    #equilibrio entre precision y recall (para clases desbalanceadas)
    f1 = f1_score(y_test, y_pred, average='macro',zero_division=0)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    conf_matrix_list = np.ndarray.tolist(conf_matrix)
    table = PrettyTable()
    table.field_names = ["", "Predicción negativa", "Predicción positiva"]
    try:
        table.add_row(["Real negativa", conf_matrix_list[0][0], conf_matrix_list[0][1]])
        table.add_row(["Real positiva", conf_matrix_list[1][0], conf_matrix_list[1][1]])
        print("Confusion Matrix:")
        print(table)
    except IndexError:
        print("La matriz de confusión contiene mas de 2 clases")

    print("\nReporte por clase:")
    #muestra un enforme con la precision, recall y f1 por clase
    print(classification_report(y_test, y_pred,zero_division=0))

    # Verificar clases no predichas nunca por el modelo
    classes = np.unique(y_test)
    #listar las clases no predichas por el modelo
    missing_preds = [cls for cls in classes if cls not in y_pred]
    if missing_preds:
        print("Clases no predichas por el modelo:", missing_preds)
    else:
        print("El modelo ha predicho todas las clases presentes en y_test.")


def evaluarEntrenamiento(model, X_train, y_train):
    #evalua el modelo sobre sus propios datos de entrenamiento
    y_pred_train = model.predict(X_train)
    #de todas las predicciones cuantas fueron correctas
    precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    #de todas las clases reales cuantas predijo
    recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    #relacion entre precision y recall
    f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

    print("\n Métricas en entrenamiento:")
    print("Precision (train):", precision)
    print("Recall (train):", recall)
    print("F1-score (train):", f1)
    print(classification_report(y_train, y_pred_train, zero_division=0))


# Programa principal
if __name__ == "__main__":

    #FASE DE ENTRENAMIENTO
    #csv_path = 'ml/casosSinteticosRiesgos.csv'
    csv_path='C:/Users/X435/Downloads/casosSinteticosRiesgos.csv'
    datos = cargar_datos(csv_path)
    print("Datos cargados correctamente. Filas:", len(datos))

    print("\nDistribución de 'evento_amenaza':")
    print(datos['evento_amenaza'].value_counts())
    print("\nPorcentaje por clase:")
    print(datos['evento_amenaza'].value_counts(normalize=True) * 100)

    #X_full, y_full, encoders = prepararDatos(datos)
    X_full, y_full = prepararDatos(datos)

    MODEL_FEATURES_PATH = "ml/model_features.joblib"
    dump(X_full.columns.tolist(), MODEL_FEATURES_PATH)
    print(f"Columnas del modelo guardadas en {MODEL_FEATURES_PATH}")

    #ENCODERS_PATH= 'ml/encoders.joblib'
    #dump(encoders, ENCODERS_PATH)
    #print(f"\nDiccionario de encoders guardado en: {ENCODERS_PATH}")

    #validacion cruzada estratificada de 5 particiones para garantizar la proporcion de clases en cada fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    y_total_true_matrix=[]
    y_total_pred_matrix=[]

    #dividimos X_full, y_full en 5 folds, en cada iteracion tenemos un conjunto de entrenamiento y otro de prueba
    for train_index, test_index in skf.split(X_full, y_full):
        print(f" Fold {fold}")
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]
        #pca = PCA(n_components=0.90, random_state=42)
        #X_train = pca.fit_transform(X_train)
        #X_test = pca.transform(X_test)
        # Aplicar SMOTE para las clases minoritarias sin alterar al resto de clases
        """print("Antes de usar SMOTE")
        print(pd.Series(y_train).value_counts())
        smote = SMOTE(
            sampling_strategy={0:50, 2:50, 3: 55},
            k_neighbors=1,
            random_state=42
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Distribución después de SMOTE:")
        print(pd.Series(y_train).value_counts())"""

        print("Antes de SMOTE:", Counter(y_train))

        """# Clase Fraude (label 3)
        FRAUDE_CLASS = 3
        TARGET_FRAUDE = 40

        if Counter(y_train)[FRAUDE_CLASS] < TARGET_FRAUDE:
            smote = SMOTE(
                sampling_strategy={FRAUDE_CLASS: TARGET_FRAUDE},
                k_neighbors=2,
                random_state=42
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("Después de SMOTE:", Counter(y_train))
        else:
            print("SMOTE no aplicado: Fraude ya tiene suficientes muestras")"""
        smote = SMOTE(
            sampling_strategy='not majority',
            k_neighbors=2,
            random_state=42
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

        #Algoritmos a evaluar
        model = modeloXGBoost()
        #model= randomForest()
        #model= knn()
        #model= svm()
        entrenamiento(model, X_train, y_train)

        print(f" Evaluación Fold {fold}")
        evaluarRendimiento(model, X_test, y_test)
        evaluarEntrenamiento(model, X_train, y_train)
        #predice eventos de amenaza con el conjunto de prueba
        y_pred = model.predict(X_test)
        print(y_pred)
        dump(model, 'ml/modelo_entrenado.joblib')
        # Guardar métricas
        #porcentaje de prediccion real (predichas reales/predichas total por el modelo)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        y_total_true_matrix.extend(y_test)
        y_total_pred_matrix.extend(y_pred)
        fold += 1

    # Resumen
    print("\nResultados promedio en validación cruzada (5 folds):")
    print(f"Accuracy promedio: {np.mean(accuracy_scores):.4f}")
    print(f"Precision promedio: {np.mean(precision_scores):.4f}")
    print(f"Recall promedio:    {np.mean(recall_scores):.4f}")
    print(f"F1-score promedio:  {np.mean(f1_scores):.4f}")

    conf_matrix_global = confusion_matrix(y_total_true_matrix, y_total_pred_matrix)

    # Obtener nombres reales de clases
    label_encoder_y = load("ml/label_encoder_y.joblib")
    class_names = label_encoder_y.inverse_transform(
        np.unique(y_total_true_matrix)
    )

    print("Confusion Matrix:")
    # -------- TABLA EN CONSOLA --------
    table = PrettyTable()
    table.field_names = ["Real \\ Pred"] + list(class_names)

    for i, row in enumerate(conf_matrix_global):
        table.add_row([class_names[i]] + list(row))

    print(table)

    """# -------- HEATMAP PROFESIONAL --------
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix_global,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicción")
    plt.ylabel("Clase Real")
    plt.title("Matriz de Confusión Global - XGBoost")
    plt.tight_layout()
    plt.show()"""


"""
# FASE PREDICCION MODELO YA ENTRENADO

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ml" / "casosSinteticosRiesgos.csv"
MODEL_PATH = BASE_DIR / "ml" / "modelo_entrenado.joblib"


def predecir():
    # 1. Cargar modelo entrenado
    model = load(MODEL_PATH)

    # 2. Leer CSV y tomar la última fila
    df = pd.read_csv(CSV_PATH, delimiter=";", encoding="cp1252")
    ultima_fila = df.index[-1]
    nueva_fila = df.tail(1).drop(columns=["evento_amenaza"], errors="ignore")

    # 3. Preprocesado igual que en entrenamiento
    nueva_fila = codificarVariablesCategoricas(nueva_fila)
    nueva_fila = tratarValoresExtremos(nueva_fila)
    nueva_fila = normalizarDatos(nueva_fila)

    # 4. Ajustar columnas a las que el modelo espera
    model_features = model.get_booster().feature_names
    nueva_fila = nueva_fila.reindex(columns=model_features, fill_value=0)

    # 5. Hacer predicción
    pred = model.predict(nueva_fila)[0]

    evento_amenaza= {
        0: "Ransomware",
        1: "DDOS (Denegación de servicio)",
        2: "Brecha de datos",
        3: "Fraude",
        4: "Destrucción física"
    }
    try:
        codigo = int(float(pred))
        etiqueta = evento_amenaza.get(codigo, f"Desconocido ({codigo})")
    except Exception:
        etiqueta = "Desconocido"

        # 7. Guardar la predicción en la última fila del CSV
    df.at[ultima_fila, "evento_amenaza"] = etiqueta
    df.to_csv(CSV_PATH, sep=";", encoding="cp1252", index=False)

    # 8. Mostrar resultado en consola (opcional)
    print(etiqueta)

if __name__ == "__main__":
    predecir()

"""