import numpy as np
import matplotlib.pyplot as plt

# Datos del problema
x = np.array([5, 10, 15, 20, 25])  # Carga en kN
y = np.array([0.6, 1.2, 1.9, 2.5, 3.1])  # Elongación en mm

# Cálculo de los coeficientes
n = len(x)
sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x * y)
sum_x2 = np.sum(x**2)

# Fórmulas de regresión lineal
b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
a = (sum_y - b * sum_x) / n

print(f"Coeficientes de la regresión:")
print(f"a (intercepto) = {a:.4f} mm")
print(f"b (pendiente) = {b:.4f} mm/kN")

# Predicción y métricas de desempeño
y_pred = a + b * x
residuals = y - y_pred
ssr = np.sum(residuals**2)  # Suma de cuadrados de los residuos
sst = np.sum((y - np.mean(y))**2)  # Suma total de cuadrados
r2 = 1 - (ssr / sst)  # Coeficiente de determinación R²

print("\nAnálisis de desempeño:")
print(f"Suma de residuos al cuadrado (SSR): {ssr:.4f}")
print(f"Coeficiente R²: {r2:.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Datos reales', markersize=8)
plt.plot(x, y_pred, '--', label=f'Ajuste lineal: y = {a:.3f} + {b:.3f}x', linewidth=2)
plt.xlabel('Carga (kN)', fontsize=12)
plt.ylabel('Elongación (mm)', fontsize=12)
plt.title('Regresión Lineal: Carga vs. Elongación', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('regresion_elongacion.png', dpi=300)
plt.show()