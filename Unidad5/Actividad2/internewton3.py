import numpy as np
import matplotlib.pyplot as plt

def newton_divided_diff(x, y):
    """Calcula la tabla de diferencias divididas de Newton"""
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    
    for j in range(1, n):
        for i in range(n - j):
            coef[i, j] = (coef[i+1, j-1] - coef[i, j-1]) / (x[i+j] - x[i])
    
    return coef[0, :]

def newton_interpolation(x_data, y_data, x):
    """Evalúa el polinomio de Newton en los puntos x"""
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
x_data = np.array([10, 20, 30, 40, 50, 60])
y_data = np.array([0.32, 0.30, 0.28, 0.27, 0.26, 0.25])

# 1. Obtener coeficientes del polinomio interpolador
coeficientes = newton_divided_diff(x_data, y_data)
print("Coeficientes del polinomio de Newton:")
for i, c in enumerate(coeficientes):
    print(f"c{i} = {c:.8f}")

# 2. Estimar Cd a V = 35 m/s
v_estimado = 35
cd_estimado = newton_interpolation(x_data, y_data, np.array([v_estimado]))
print(f"\nCoeficiente de arrastre estimado a V = {v_estimado} m/s: {cd_estimado[0]:.4f}")

# 3. Generar gráfica
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_interp = newton_interpolation(x_data, y_data, x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', markersize=8, label='Datos reales')
plt.plot(x_vals, y_interp, 'b-', linewidth=2, label='Interpolación de Newton')
plt.xlabel('Velocidad del aire (m/s)', fontsize=12)
plt.ylabel('Coeficiente de arrastre ($C_d$)', fontsize=12)
plt.legend()
plt.title('Interpolación de Newton para el coeficiente de arrastre', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x_data)
plt.savefig("newton_interpolacion_arrastre.png")
plt.show()

