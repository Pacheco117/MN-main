import numpy as np
import matplotlib.pyplot as plt

# Datos del problema
x = np.array([0, 2, 4, 6, 8])    # Posición en cm
y = np.array([100, 92, 85, 78, 71])  # Temperatura en °C

# Cálculo de coeficientes
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

# Fórmulas de regresión lineal
b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a = (sum_y - b * sum_x) / n

print("Coeficientes de la regresión:")
print(f"a (intercepto) = {a:.2f} °C")
print(f"b (pendiente) = {b:.2f} °C/cm\n")

# Predicción en x=5 cm
x_pred = 5
y_pred = a + b * x_pred
print(f"Temperatura estimada en x=5 cm: {y_pred:.1f} °C")

# Métricas de desempeño
y_modelo = a + b * x
residuals = y - y_modelo
ssr = np.sum(residuals**2)
sst = np.sum((y - np.mean(y))**2)
r2 = 1 - (ssr / sst)

print("\nAnálisis de desempeño:")
print(f"Suma de residuos al cuadrado (SSR): {ssr:.2f}")
print(f"Coeficiente R²: {r2:.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, y, 's', label='Datos reales', markersize=8, color='navy')
plt.plot(x, y_modelo, '--', label=f'y = {a:.1f} + {b:.1f}x', linewidth=2, color='crimson')
plt.scatter(x_pred, y_pred, color='red', zorder=10, label=f'Predicción en x=5 cm: {y_pred:.1f}°C')
plt.xlabel('Posición (cm)', fontsize=12)
plt.ylabel('Temperatura (°C)', fontsize=12)
plt.title('Regresión Lineal: Disminución de Temperatura', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('transferencia_calor.png', dpi=300)
plt.show()