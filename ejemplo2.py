import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC # Importamos el clasificador SVM

# 1. Generar datos linealmente inseparables en 2D
np.random.seed(42)

num_inner = 70
theta_inner = 2 * np.pi * np.random.rand(num_inner)
r_inner = 1.0 + np.random.randn(num_inner) * 0.2
X_inner_raw = np.c_[r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)]
X_inner = np.abs(X_inner_raw) + 0.1
y_inner = np.zeros(num_inner)

num_outer = 90
theta_outer = 2 * np.pi * np.random.rand(num_outer)
r_outer = 2.5 + np.random.randn(num_outer) * 0.3
X_outer_raw = np.c_[r_outer * np.cos(theta_outer), r_outer * np.sin(theta_outer)]
X_outer = np.abs(X_outer_raw) + 0.1
y_outer = np.ones(num_outer)

X_2d = np.vstack((X_inner, X_outer))
y = np.hstack((y_inner, y_outer))

# 2. Visualizar los datos en 2D
plt.figure(figsize=(9, 7))
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], color='red', label='Clase 0 (Rojo)')
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], color='green', label='Clase 1 (Verde)')
plt.title('Datos Linealmente Inseparables en 2D')
plt.xlabel('Dosis Componente A')
plt.ylabel('Dosis Componente B')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()

# 3. Transformación no lineal (Elevación a 3D usando la distancia cuadrática al origen)
X_3d_transformed = np.c_[X_2d, X_2d[:, 0]**2 + X_2d[:, 1]**2]

svm_3d = SVC(kernel='linear', C=1000) # C muy alto para una separación "dura"
svm_3d.fit(X_3d_transformed, y)

coef = svm_3d.coef_[0]
intercept = svm_3d.intercept_[0]

x_min, x_max = X_3d_transformed[:, 0].min(), X_3d_transformed[:, 0].max()
y_min, y_max = X_3d_transformed[:, 1].min(), X_3d_transformed[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                     np.linspace(y_min, y_max, 50))

if np.isclose(coef[2], 0):
    print("Advertencia: El coeficiente para Z es muy cercano a cero. El hiperplano podría ser casi vertical.")
    # Manejar el caso si es necesario, por ejemplo, mostrando un plano vertical.
    zz = np.full_like(xx, -intercept / coef[2] if not np.isclose(coef[2], 0) else np.nan) # Valor constante si coef[2] es 0
else:
    zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

# 4. Visualizar los datos transformados en 3D y el hiperplano
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos transformados
ax.scatter(X_3d_transformed[y == 0, 0], X_3d_transformed[y == 0, 1], X_3d_transformed[y == 0, 2],
           color='red', label='Clase 0 (Rojo)', s=50) # Aumenté el tamaño para visibilidad
ax.scatter(X_3d_transformed[y == 1, 0], X_3d_transformed[y == 1, 1], X_3d_transformed[y == 1, 2],
           color='green', label='Clase 1 (Verde)', s=50) # Aumenté el tamaño para visibilidad

# Graficar el hiperplano
ax.plot_surface(xx, yy, zz, alpha=0.5, color='blue', label='Hiperplano de Separación') # Alpha para transparencia
ax.view_init(elev=20, azim=-120) # Ajusta el ángulo de vista para mejor perspectiva

ax.set_title('Datos Transformados a 3D con Hiperplano de Separación SVM')
ax.set_xlabel('Dosis Componente A (x)')
ax.set_ylabel('Dosis Componente B (y)')
ax.set_zlabel('Distancia Cuadrática al Origen (z)')
ax.legend()
plt.show()

# 5. Calcular y mostrar la Matriz de Kernel RBF (Gaussiano)
kernel_matrix_rbf = rbf_kernel(X_2d, gamma=0.1)

print("\n--- Matriz de Kernel RBF (Gaussiano) para el Ejemplo 2---")
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print(kernel_matrix_rbf)