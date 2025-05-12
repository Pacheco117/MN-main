import numpy as np
import matplotlib.pyplot as plt

# Datos del problema
x = np.array([50, 70, 90, 110, 130])  # Presión en kPa
y = np.array([15, 21, 27, 33, 39])     # Caudal en L/min

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
print(f"a (intercepto) = {a:.2f} L/min")
print(f"b (pendiente) = {b:.2f} L/(min·kPa)\n")

# Predicción en x=100 kPa
x_pred = 100
y_pred = a + b * x_pred
print(f"Caudal estimado a 100 kPa: {y_pred:.1f} L/min")

# Métricas de desempeño (corregido)
y_modelo = a + b * x
residuals = y - y_modelo
ssr = np.sum(residuals**2)
sst = np.sum((y - np.mean(y))**2)  # ✅ Paréntesis corregido
r2 = 1 - (ssr / sst)

print("\nAnálisis de desempeño:")
print(f"Suma de residuos al cuadrado (SSR): {ssr:.2f}")
print(f"Coeficiente R²: {r2:.4f}")

# Gráfica
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Datos reales', markersize=8, color='darkgreen')
plt.plot(x, y_modelo, '--', label=f'y = {a:.1f} + {b:.2f}x', linewidth=2, color='orange')
plt.scatter(x_pred, y_pred, color='red', zorder=10, label=f'Predicción en x=100 kPa: {y_pred:.1f} L/min')
plt.xlabel('Presión (kPa)', fontsize=12)
plt.ylabel('Caudal (L/min)', fontsize=12)
plt.title('Regresión Lineal: Presión vs. Caudal', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('caudal_tuberia.png', dpi=300)
plt.show()