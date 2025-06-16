import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.svm import SVC # Importamos el clasificador SVM

# 1. Generar datos linealmente inseparables en 1D (solo positivos)
# Los puntos verdes estarán en un rango intermedio de dosis positivas.
# Los puntos rojos estarán en dosis más bajas y más altas.
np.random.seed(42) # Para reproducibilidad

# Dosis bajas (rojos)
X_low_dosage = np.random.uniform(0.5, 2.0, 4).reshape(-1, 1)
y_low = np.zeros(len(X_low_dosage))

# Dosis intermedias (verdes)
X_mid_dosage = np.random.uniform(3.0, 5.0, 5).reshape(-1, 1)
y_mid = np.ones(len(X_mid_dosage))

# Dosis altas (rojos)
X_high_dosage = np.random.uniform(6.0, 7.5, 4).reshape(-1, 1)
y_high = np.zeros(len(X_high_dosage))

X_1d = np.vstack((X_low_dosage, X_mid_dosage, X_high_dosage))
y = np.hstack((y_low, y_mid, y_high))

# Ordenar los datos por dosis para una mejor visualización en 1D
sorted_indices = np.argsort(X_1d[:, 0])
X_1d_sorted = X_1d[:][sorted_indices]
y_sorted = y[:][sorted_indices]

# 2. Visualizar los datos en 1D (solo positivos)
plt.figure(figsize=(8, 4))
plt.scatter(X_1d_sorted[(y_sorted == 0)], np.zeros(np.sum(y_sorted == 0)), color='red', label='Clase 0 (Rojo)')
plt.scatter(X_1d_sorted[(y_sorted == 1)], np.zeros(np.sum(y_sorted == 1)), color='green', label='Clase 1 (Verde)')
plt.title('Datos Linealmente Inseparables en 1D')
plt.xlabel('Dosage (mg)')
plt.yticks([])
plt.grid(True)
plt.legend()
plt.show() # Muestra la primera gráfica

# 3. Transformación no lineal (función cuadrática: y = x^2)
# Mapeamos x a (x, x^2)
X_2d_transformed = np.c_[X_1d, X_1d**2]

# Ordenar los datos transformados por la dosis original para visualización
X_2d_transformed_sorted = X_2d_transformed[:][sorted_indices]

# --- NUEVO CÓDIGO PARA ENTRENAR SVM Y MOSTRAR HIPERPLANO ---

# Entrenar un SVM lineal en el espacio transformado 2D
# Usamos un C alto porque los datos son claramente linealmente separables en 2D después de la transformación
svm_2d = SVC(kernel='linear', C=1000)
svm_2d.fit(X_2d_transformed, y)

# Obtener los coeficientes de la línea: w_0*x + w_1*y + intercept = 0
# Aquí x es X_2d_transformed[:, 0] y y es X_2d_transformed[:, 1]
w = svm_2d.coef_[0]
b = svm_2d.intercept_[0]

# Crear puntos para dibujar la línea de separación
# Vamos a calcular los valores de y (la segunda dimensión) en función de x (la primera dimensión)
# Linea: y = (-w[0]*x - b) / w[1]
x_line = np.linspace(X_2d_transformed[:, 0].min() - 0.5, X_2d_transformed[:, 0].max() + 0.5, 100)
y_line = (-w[0] * x_line - b) / w[1]

# 4. Visualizar los datos transformados en 2D con el hiperplano (línea)
plt.figure(figsize=(8, 6))
plt.scatter(X_2d_transformed_sorted[(y_sorted == 0), 0], X_2d_transformed_sorted[(y_sorted == 0), 1],
            color='red', label='Clase 0 (Rojo)')
plt.scatter(X_2d_transformed_sorted[(y_sorted == 1), 0], X_2d_transformed_sorted[(y_sorted == 1), 1],
            color='green', label='Clase 1 (Verde)')

# Dibujar la línea de separación (hiperplano)
plt.plot(x_line, y_line, color='blue', linestyle='--', label='Hiperplano de Separación SVM')

plt.title('Datos Transformados a 2D con Hiperplano de Separación Lineal')
plt.xlabel('Dosage (mg)')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.ylim(X_2d_transformed[:, 1].min() - 1, X_2d_transformed[:, 1].max() + 1) # Asegurar que la línea se vea bien
plt.xlim(X_2d_transformed[:, 0].min() - 0.5, X_2d_transformed[:, 0].max() + 0.5)
plt.show() # Muestra la segunda gráfica con el hiperplano

kernel_matrix = polynomial_kernel(X_1d, degree=2, coef0=0.5)

print("\n--- Matriz de Kernel Polinomial para el Ejemplo 1")
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print(kernel_matrix)