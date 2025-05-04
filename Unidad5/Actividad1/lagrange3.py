import numpy as np
import matplotlib.pyplot as plt

# Puntos de interpolación del problema
x_points = np.array([2.0, 4.0, 6.0, 8.0])
y_points = np.array([2500, 2300, 2150, 2050])

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

# Cálculo del consumo en x = 5.0 km
x_eval = 5.0
consumo = lagrange_interpolation(x_eval, x_points, y_points)
print(f"Consumo en x = {x_eval} km: {consumo:.2f} kg/h")
# Crear puntos para la gráfica
x_values = np.linspace(min(x_points), max(x_points), 100)
y_values = [lagrange_interpolation(x, x_points, y_points) for x in x_values]

# Configurar la gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Interpolación de Lagrange", color="purple")
plt.scatter(x_points, y_points, color="red", label="Datos originales", zorder=5)
plt.xlabel("Altitud (km)", fontsize=12)
plt.ylabel("Consumo (kg/h)", fontsize=12)
plt.title("Interpolación de Lagrange para el Consumo de Combustible en Aeronaves", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("consumo_avion.png")
plt.show()