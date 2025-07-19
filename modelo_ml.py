# Librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from prettytable import PrettyTable
import psutil
import os
import time
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
    return df.drop(columnas_unicas, axis=1)

def missingValues(df):
    return df.isnull().any().any()

def fillMissingValues(df):
    df.fillna(df.mean(), inplace=True)
    return df


def dibujarMatrizCorrelación(df):
    # Selecciona solo las columnas numéricas
    df_num = df.select_dtypes(include=[np.number])
    corr = df_num.corr()

    plt.figure(figsize=(40, 40))
    sns.heatmap(corr, annot=True)
    plt.savefig("correlation_matrix.svg", format='svg')
    plt.show()


def eliminarColumnasRedundantes(df):
  # Calcular la matriz de correlación
  corr_matrix = df.corr().abs()

  # Seleccionar la diagonal superior de la matriz de correlación
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

  # Encontrar columnas con una correlación mayor a un umbral
  to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]

  # Elimina las columnas seleccionadas
  return df.drop(df[to_drop], axis=1)

def codificarVariablesCategoricas(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.factorize(df[col])[0]
    return df
def hayValoresExtremos(df):
  mask = np.isinf(df) | np.isnan(df) | (df > 1e5)
  return mask.sum().sum() > 0
def tratarValoresExtremos(df):
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df


def normalizarDatos(df):
    if 'evento_amenaza' in df.columns:
        target = df['evento_amenaza']
        df = df.drop('evento_amenaza', axis=1)
    else:
        target = None

    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled_df, columns=df.columns)

    if target is not None:
        df_scaled['evento_amenaza'] = target

    return df_scaled
def analisisPCA(df):
  explained_pca_ratio = 0.90
  #print(df.columns)
  data_scaled = pd.DataFrame(preprocessing.scale(df), columns=df.columns)

  pca = PCA().fit(data_scaled)
  line_data = np.cumsum(pca.explained_variance_ratio_)
  line_data = np.insert(line_data, 0, 0)
  plt.bar(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='g')
  plt.plot(np.arange(0, len(line_data)), line_data, marker='D')
  plt.xlim(0, len(pca.explained_variance_ratio_), 1)
  plt.axhline(y=explained_pca_ratio, color='black', linestyle='--')
  plt.xlabel('Number of components')
  plt.ylabel('Cumulative explained variance')
  sklearn_pca = PCA(n_components=explained_pca_ratio)
  return sklearn_pca.fit_transform(data_scaled)
def kernelPca(df):
  kpca = KernelPCA(n_components=18, kernel='rbf') # en este caso, reducimos las 84 columnas a 18 componentes principales y utilizamos un kernel gaussiano (rbf)

  # Aplica KernelPCA al dataframe
  kpca_components = kpca.fit_transform(df)

  # Crea un nuevo dataframe con las componentes principales
  df_kpca = pd.DataFrame(kpca_components, columns=df.columns)
  print(df_kpca)
def prepararDatos(df):
    print("Eliminando columnas vacías...")
    df = eliminarColumnasVacias(df)

    print("Rellenando valores vacíos...")
    if missingValues(df):
        df = fillMissingValues(df)

    print("Codificando variables categóricas...")
    df = codificarVariablesCategoricas(df)

    print("Dibujando matriz de correlación...")
    dibujarMatrizCorrelación(df)

    print("Eliminando columnas redundantes...")
    df = eliminarColumnasRedundantes(df)

    print("Tratando valores extremos...")
    df = tratarValoresExtremos(df)

    print("Normalizando datos...")
    df = normalizarDatos(df)

    if 'evento_amenaza' not in df.columns:
        raise ValueError("La columna 'evento_amenaza' no está presente en los datos.")

    X = df.drop(['evento_amenaza'], axis=1)
    y = df['evento_amenaza']

    print("Número de columnas resultantes: ", df.shape[1], "\n")
    print(df.columns)

    return X, y

#Algoritmos de prediccion
def modeloXGBoost():
    return XGBClassifier(
        objective='multi:softprob',
        num_class=5,
        learning_rate=0.05,
        max_depth=2,
        n_estimators=50,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss'
    )

def randomForest():
    return RandomForestClassifier(criterion='entropy', max_depth=10, max_features=10, class_weight='balanced')

def knn():
  return KNeighborsClassifier(n_neighbors=35, algorithm='kd_tree', leaf_size=20, p=1)
def svm():
    return SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        probability=True,
        random_state=42
    )

#Entrenamiento
def entrenamiento(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def evaluarRendimiento(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred,average='weighted',zero_division=0)
    recall = recall_score(y_test, y_pred,average='weighted',zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted',zero_division=0)

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
    print(classification_report(y_test, y_pred,zero_division=0))

    # Verificar clases no predichas
    classes = np.unique(y_test)
    missing_preds = [cls for cls in classes if cls not in y_pred]
    if missing_preds:
        print("Clases no predichas por el modelo:", missing_preds)
    else:
        print("El modelo ha predicho todas las clases presentes en y_test.")


def evaluarEntrenamiento(model, X_train, y_train):
    y_pred_train = model.predict(X_train)
    precision = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
    recall = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
    f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)

    print("\n Métricas en entrenamiento:")
    print("Precision (train):", precision)
    print("Recall (train):", recall)
    print("F1-score (train):", f1)
    print(classification_report(y_train, y_pred_train, zero_division=0))


# Programa principal
if __name__ == "__main__":
    csv_path = 'C:/Users/X435/Downloads/casosSinteticosRiesgos.csv'
    datos = cargar_datos(csv_path)
    print("Datos cargados correctamente. Filas:", len(datos))

    print("\nDistribución de 'evento_amenaza':")
    print(datos['evento_amenaza'].value_counts())
    print("\nPorcentaje por clase:")
    print(datos['evento_amenaza'].value_counts(normalize=True) * 100)

    X_full, y_full = prepararDatos(datos)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X_full, y_full):
        print(f" Fold {fold}")
        X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train, y_test = y_full.iloc[train_index], y_full.iloc[test_index]

        # Aplicar SMOTE
        #smote = SMOTE(random_state=42)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        smote = SMOTE(
            sampling_strategy={3: 60, 4: 50},  # Ajusta según necesidad
            random_state=42
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        model = modeloXGBoost()
        #model= knn()
        #model= randomForest()
        #model= svm()
        entrenamiento(model, X_train, y_train)

        print(f" Evaluación Fold {fold}")
        evaluarRendimiento(model, X_test, y_test)
        evaluarEntrenamiento(model, X_train, y_train)

        y_pred = model.predict(X_test)

        # Guardar métricas
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        fold += 1

    # Resumen
    print("\nResultados promedio en validación cruzada (5 folds):")
    print(f"Accuracy promedio: {np.mean(accuracy_scores):.4f}")
    print(f"Precision promedio: {np.mean(precision_scores):.4f}")
    print(f"Recall promedio:    {np.mean(recall_scores):.4f}")
    print(f"F1-score promedio:  {np.mean(f1_scores):.4f}")