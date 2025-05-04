import numpy as np
import matplotlib.pyplot as plt

# Puntos de interpolación del problema
x_points = np.array([1.0, 2.5, 4.0, 5.5])
y_points = np.array([85, 78, 69, 60])

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

# Cálculo de la temperatura en x = 3.0 cm
x_eval = 3.0
temperatura = lagrange_interpolation(x_eval, x_points, y_points)
print(f"Temperatura en x = {x_eval} cm: {temperatura:.2f} °C")
# Crear puntos para la gráfica
x_values = np.linspace(min(x_points), max(x_points), 100)
y_values = [lagrange_interpolation(x, x_points, y_points) for x in x_values]

# Configurar la gráfica
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label="Interpolación de Lagrange", color="green")
plt.scatter(x_points, y_points, color="orange", label="Datos originales", zorder=5)
plt.xlabel("Profundidad (cm)", fontsize=12)
plt.ylabel("Temperatura (°C)", fontsize=12)
plt.title("Interpolación de Lagrange para la Temperatura en un Motor", fontsize=14)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("temperatura_motor.png")
plt.show()