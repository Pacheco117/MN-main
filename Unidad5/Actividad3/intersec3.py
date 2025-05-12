import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Datos de consumo de combustible
datos_x = np.array([40, 60, 80, 100, 120, 140])
datos_y = np.array([6.5, 5.8, 5.5, 5.7, 6.2, 7.0])

# Interpolaciones segmentadas
lineal_interp = interp1d(datos_x, datos_y, kind='linear')
cuadratica_interp = interp1d(datos_x, datos_y, kind='quadratic')
cubica_interp = interp1d(datos_x, datos_y, kind='cubic')

# Valores para graficar
x_vals = np.linspace(40, 140, 100)
y_lineal = lineal_interp(x_vals)
y_cuadratica = cuadratica_interp(x_vals)
y_cubica = cubica_interp(x_vals)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(datos_x, datos_y, color='red', label='Datos originales', zorder=5)
plt.plot(x_vals, y_lineal, '--', label='Interpolación Lineal', color='blue')
plt.plot(x_vals, y_cuadratica, '-.', label='Interpolación Cuadrática', color='green')
plt.plot(x_vals, y_cubica, label='Interpolación Cúbica', color='purple')
plt.xlabel('Velocidad (km/h)')
plt.ylabel('Consumo (L/100 km)')
plt.title('Comparación de Métodos de Interpolación para el Consumo de Combustible')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('consumo_combustible.png')
plt.show()