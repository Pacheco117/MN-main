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
x_data = np.array([50, 100, 150, 200])
y_data = np.array([0.12, 0.35, 0.65, 1.05])

# 1. Obtener coeficientes del polinomio
coeficientes = newton_divided_diff(x_data, y_data)
print("Coeficientes del polinomio de Newton:")
print("c0 =", coeficientes[0])
print("c1 =", coeficientes[1])
print("c2 =", coeficientes[2])
print("c3 =", coeficientes[3])

# 2. Estimar deformación para 125 N
f_estimado = 125
epsilon_estimado = newton_interpolation(x_data, y_data, np.array([f_estimado]))
print(f"\nDeformación estimada para F = {f_estimado} N: {epsilon_estimado[0]:.4f} mm")

# 3. Generar gráfica
x_vals = np.linspace(min(x_data), max(x_data), 100)
y_interp = newton_interpolation(x_data, y_data, x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_data, y_data, 'ro', label='Datos originales')
plt.plot(x_vals, y_interp, 'b-', label='Interpolación de Newton')
plt.xlabel('Carga aplicada (N)')
plt.ylabel('Deformación (mm)')
plt.legend()
plt.title('Interpolación de Newton para la deformación del material')
plt.grid(True)
plt.savefig("newton_interpolacion.png")
plt.show()
