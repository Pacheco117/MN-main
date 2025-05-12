import numpy as np
import matplotlib.pyplot as plt

def newton_divided_diff(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    coef = newton_divided_diff(x_data, y_data)
    n = len(x_data)
    
    y_interp = np.zeros_like(x)
    for i in range(len(x)):
        term = coef[0]
        product = 1
        for j in range(1, n):
            product *= (x[i] - x_data[j-1])
            term += coef[j] * product
        y_interp[i] = term
    
    return y_interp

# Datos del problema
x_data = np.array([200, 250, 300, 350, 400])
y_data = np.array([30, 35, 40, 46, 53])

# 1. Construir el polinomio interpolador
coeficientes = newton_divided_diff(x_data, y_data)
print("Coeficientes del polinomio de Newton:")
for i in range(len(coeficientes)):
    print(f"c{i} = {coeficientes[i]:.8f}")

# 2. Predecir eficiencia a T = 275°C
t_estimado = 275
eficiencia_estimada = newton_interpolation(x_data, y_data, np.array([t_estimado]))
print(f"\nEficiencia estimada a T = {t_estimado}°C: {eficiencia_estimada[0]:.2f}%")

# 3. Generar gráfica
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_interp = newton_interpolation(x_data, y_data, x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', markersize=8, label='Datos experimentales')
plt.plot(x_vals, y_interp, 'b-', linewidth=2, label='Interpolación de Newton')
plt.xlabel('Temperatura de entrada (°C)', fontsize=12)
plt.ylabel('Eficiencia (%)', fontsize=12)
plt.legend()
plt.title('Interpolación de Newton para la eficiencia del motor térmico', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x_data)
plt.savefig("newton_interpolacion_motor.png")
plt.show()
