import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score # ¡Importamos KFold y cross_val_score!
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)
from sklearn.datasets import make_classification
import seaborn as sns
from scipy import stats # Para la prueba t de significancia estadística

# --- Global Definitions (Constants) ---
CLASS_NAMES_MAP = {0: 'Small', 1: 'Medium', 2: 'Big'}
FEATURE_NAMES = ['Height (cm)', 'Weight (kg)', 'Mouth Length (cm)', 'Color (Scale 0-1)']
N_FOLDS = 10 # Definimos el número de folds para la validación cruzada

# --- Functions ---

def generate_data(n_samples=int(math.pow(10, 3)), n_features=4, n_informative=2,
                  n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1,
                  flip_y=0.1, class_sep=0.3, random_state=42):
    
    
    """
    Genera un conjunto de datos de clasificación sintético.

    Args:
        n_samples (int): Número total de muestras.
        n_features (int): Número total de características.
        n_informative (int): Número de características informativas.
        n_redundant (int): Número de características redundantes.
        n_repeated (int): Número de características repetidas.
        n_classes (int): Número de clases a generar.
        n_clusters_per_class (int): Número de clústeres por clase.
        flip_y (float): Fracción de etiquetas a invertir aleatoriamente.
        class_sep (float): Separación entre las clases.
        random_state (int): Semilla para la reproducibilidad.

    Returns:
        tuple: (X, y) donde X son las características y y son las etiquetas.
    """
    print("--- Generando Conjunto de Datos Sintético ---")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=random_state
    )

    # Asegurar que las características sean no negativas
    X = X - X.min()
    # print(f"Forma de X (Características): {X.shape}")
    # print(f"Forma de y (Etiquetas numéricas originales): {y.shape}")
    # print(f"Conteo de clases en y (numérico): {np.bincount(y)}")

    # No es necesario X_df ni y_labeled para el flujo de cross-validation
    # pero puedes usarlos si quieres inspeccionar el DataFrame o etiquetas con nombres.
    # X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    # y_labeled = pd.Series(y).map(CLASS_NAMES_MAP)

    return X, y

def display_confusion_matrix(cm, title, class_names_map):
    """
    Muestra una matriz de confusión usando Seaborn Heatmap.
    Nota: Con cross-validation, no obtenemos una única CM.
    Esta función se mantendría para el caso de un solo train/test split.
    """
    print(f"\nNota: La Matriz de Confusión no se visualiza directamente en la Validacion Cruzada.")
    print(f"Los resultados son promedios de {N_FOLDS} folds.")
    # Si quieres una CM, tendrías que entrenar el modelo una vez con X_train/y_train y predecir sobre X_test/y_test
    # como se hacía en tu código original, o promediar CMs de los folds (más complejo).
    # En este contexto, nos centramos en las métricas promedio.

def print_model_metrics(model_name, accuracy_scores, f1_scores, roc_auc_scores):
    """
    Calcula y muestra las métricas de rendimiento promedio y desviación estándar.

    Args:
        model_name (str): Nombre del modelo para imprimir en la salida.
        accuracy_scores (np.array): Scores de exactitud de cada fold.
        f1_scores (np.array): Scores de F1 de cada fold.
        roc_auc_scores (np.array): Scores de AUC-ROC de cada fold.
    """
    avg_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_roc_auc = np.mean(roc_auc_scores)
    std_roc_auc = np.std(roc_auc_scores)

    # print(f"\n--- Métricas de Rendimiento del Modelo {model_name} (Cross-Validation K={N_FOLDS}) ---")
    # print(f"Exactitud (Accuracy) - {model_name}: {avg_accuracy:.4f} (+/- {std_accuracy:.4f})")
    # print(f"Puntuación F1 (F1-Score, ponderado) - {model_name}: {avg_f1:.4f} (+/- {std_f1:.4f})")
    # print(f"Área bajo la curva ROC (AUC-ROC, ponderado OVR) - {model_name}: {avg_roc_auc:.4f} (+/- {std_roc_auc:.4f})")

    # print(f"\n--- Interpretación del Modelo {model_name} ---")
    # print(f"Con una Exactitud promedio de {avg_accuracy:.4f}, el modelo {model_name} tuvo un rendimiento general de {avg_accuracy*100:.2f}% de aciertos.")
    # if avg_accuracy < 0.6:
    #     print("Parece que los datos no son tan linealmente separables o el modelo tiene dificultades para generalizar.")
    # elif avg_accuracy < 0.8:
    #     print("El rendimiento del modelo es moderado. Podría mejorar con el ajuste fino de sus hiperparámetros.")
    # else:
    #     print("El rendimiento del modelo es bueno. Sugiere que las fronteras de decisión son adecuadas para estos datos.")

    return {
        'name': model_name,
        'accuracy_scores': accuracy_scores, # Devolvemos los scores individuales para la prueba t
        'f1_scores': f1_scores,
        'roc_auc_scores': roc_auc_scores,
        'avg_accuracy': avg_accuracy
    }


def train_and_evaluate_model_cv(model, X, y, model_name, n_folds=N_FOLDS, random_state=42):
    """
    Entrena y evalúa un modelo usando Validación Cruzada K-Fold.

    Args:
        model (estimator): El objeto modelo de Scikit-learn (ej., LogisticRegression, SVC).
        X (np.array): Características del dataset completo.
        y (np.array): Etiquetas del dataset completo.
        model_name (str): Nombre descriptivo del modelo.
        n_folds (int): Número de folds para la validación cruzada.
        random_state (int): Semilla para la reproducibilidad de KFold.

    Returns:
        dict: Un diccionario con las métricas promedio del modelo y los scores por fold.
    """
    print(f"\n--- Evaluando Modelo: {model_name} con Validación Cruzada K={n_folds} ---")

    # KFold para generar las divisiones de forma reproducible
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Obtenemos los scores de exactitud para cada fold
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_weighted')
    roc_auc_scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc_ovr_weighted')

    results = print_model_metrics(model_name, accuracy_scores, f1_scores, roc_auc_scores)
    return results

def print_final_comparison(logistic_results, svm_rbf_results):
    """
    Imprime un resumen comparativo de los dos modelos y realiza una prueba de significancia.

    Args:
        logistic_results (dict): Métricas y scores del modelo de Regresión Logística.
        svm_rbf_results (dict): Métricas y scores del modelo SVM RBF.
    """
    print("\n" + "="*70)
    print("     FINAL COMPARATIVE SUMMARY OF CLASSIFIERS (CV AVERAGES)      ")
    print("="*70)

    print(f"\nLogistic Regression (Linear Classifier):")
    print(f"  Average Accuracy: {logistic_results['avg_accuracy']:.4f}")
    print(f"  Average F1-Score: {np.mean(logistic_results['f1_scores']):.4f}")
    print(f"  Average AUC-ROC: {np.mean(logistic_results['roc_auc_scores']):.4f}")

    print(f"\nSVM with RBF Kernel (Non-Linear Classifier):")
    print(f"  Average Accuracy: {svm_rbf_results['avg_accuracy']:.4f}")
    print(f"  Average F1-Score: {np.mean(svm_rbf_results['f1_scores']):.4f}")
    print(f"  Average AUC-ROC: {np.mean(svm_rbf_results['roc_auc_scores']):.4f}")


    # Comentario sobre la mejora
    if svm_rbf_results['avg_accuracy'] > logistic_results['avg_accuracy']:
        diff = svm_rbf_results['avg_accuracy'] - logistic_results['avg_accuracy']
        print(f"\n¡En promedio, el SVM con Kernel RBF superó a la Regresión Logística en Exactitud por {diff:.4f} puntos!")
        print("Esto sugiere que una frontera de decisión no lineal es más adecuada para la estructura de estos datos.")
    else:
        diff = logistic_results['avg_accuracy'] - svm_rbf_results['avg_accuracy']
        print(f"\nEn promedio, la Regresión Logística fue superior (o muy similar) al SVM con Kernel RBF en Exactitud por {diff:.4f} puntos.")
        print("Esto podría indicar que los datos son predominantemente linealmente separables, o que el SVM RBF necesita un ajuste fino de hiperparámetros.")

    # --- Prueba de Significancia Estadística (t-test pareada) ---
    print("\n--- Realizando Prueba de Significancia Estadística (t-test pareada) ---")
    # Comparamos los arrays de scores de exactitud de cada fold
    t_statistic, p_value = stats.ttest_rel(logistic_results['accuracy_scores'], svm_rbf_results['accuracy_scores'])

    print(f"Estadístico t (Exactitud): {t_statistic:.4f}")
    print(f"Valor p (Exactitud): {p_value:.4f}")

    alpha = 0.05 # Nivel de significancia

    if p_value < alpha:
        print(f"\nDado que el valor p ({p_value:.4f}) es menor que el nivel de significancia ({alpha}),")
        print("la diferencia promedio en Exactitud entre los modelos es **estadísticamente significativa**.")
        if svm_rbf_results['avg_accuracy'] > logistic_results['avg_accuracy']:
            print("El **SVM RBF es estadísticamente superior** a la Regresión Logística en este problema.")
        else:
            print("La **Regresión Logística es estadísticamente superior** al SVM RBF en este problema.")
    else:
        print(f"\nDado que el valor p ({p_value:.4f}) es mayor o igual que el nivel de significancia ({alpha}),")
        print("la diferencia promedio en Exactitud entre los modelos **NO es estadísticamente significativa**.")
        print("Esto sugiere que cualquier diferencia observada podría deberse al azar y no a una superioridad inherente.")

    print("\n" + "="*70)


# --- Main Execution Block ---
def main():
    """
    Función principal para ejecutar el flujo completo del análisis de clasificación con Validación Cruzada.
    """
    # 1. Generar los datos (X, y completos)
    X, y = generate_data()

    # 2. No necesitamos split_data() aquí, cross_val_score lo hace internamente.
    #    Pero si quieres un train/test split para visualizar la CM de una única corrida,
    #    puedes descomentar las líneas relacionadas y adaptar display_confusion_matrix.
    # X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. Entrenar y evaluar Regresión Logística con Cross-Validation
    model_logistic = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
    logistic_results = train_and_evaluate_model_cv(model_logistic, X, y, "Regresión Logística (Lineal)")

    # 4. Entrenar y evaluar SVM con Kernel RBF con Cross-Validation
    model_svm_rbf = SVC(kernel='rbf', random_state=42, probability=True)
    svm_rbf_results = train_and_evaluate_model_cv(model_svm_rbf, X, y, "SVM (Kernel RBF)")

    # 5. Comparar resultados y realizar prueba de significancia
    print_final_comparison(logistic_results, svm_rbf_results)

if __name__ == "__main__":
    main()