import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Datos de deflexión de la viga
datos_x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
datos_y = np.array([0.0, -1.5, -2.8, -3.0, -2.7, -2.0])

# Interpolaciones segmentadas
lineal_interp = interp1d(datos_x, datos_y, kind='linear')
cuadratica_interp = interp1d(datos_x, datos_y, kind='quadratic')
cubica_interp = interp1d(datos_x, datos_y, kind='cubic')

# Valores para graficar
x_vals = np.linspace(0, 5, 100)
y_lineal = lineal_interp(x_vals)
y_cuadratica = cuadratica_interp(x_vals)
y_cubica = cubica_interp(x_vals)

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.scatter(datos_x, datos_y, color='red', label='Datos originales', zorder=5)
plt.plot(x_vals, y_lineal, '--', label='Interpolación Lineal', color='blue')
plt.plot(x_vals, y_cuadratica, '-.', label='Interpolación Cuadrática', color='green')
plt.plot(x_vals, y_cubica, label='Interpolación Cúbica', color='purple')
plt.xlabel('Longitud (m)')
plt.ylabel('Deflexión (mm)')
plt.title('Comparación de Métodos de Interpolación Segmentada')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('deflexion_viga.png')
plt.show()