import numpy as np
import matplotlib.pyplot as plt

# Puntos de interpolación del problema
x_points = np.array([0.5, 1.0, 1.5, 2.0])
y_points = np.array([1.2, 2.3, 3.7, 5.2])

# Función de interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Cálculo de la deformación en x = 1.25 m
x_eval = 1.25
deformacion = lagrange_interpolation(x_eval, x_points, y_points)
print(f"Deformación en x = {x_eval} m: {deformacion:.3f} mm")
# Crear puntos para la gráfica
x_values = np.linspace(min(x_points), max(x_points), 100)
y_values = [lagrange_interpolation(x, x_points, y_points) for x in x_values]

# Configurar la gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Interpolación de Lagrange", color="blue")
plt.scatter(x_points, y_points, color="red", label="Datos originales")
plt.xlabel("Posición (m)", fontsize=12)
plt.ylabel("Deformación (mm)", fontsize=12)
plt.title("Interpolación de Lagrange para la Deformación de una Viga", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("deformacion_viga.png")
plt.show()