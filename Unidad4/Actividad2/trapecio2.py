import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Función a integrar: f(x) = e^{-x²}
def f(x):
    return np.exp(-x**2)

# Implementación de la regla del trapecio compuesta
def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])  # Corregido
    return integral, x, y

# Parámetros de integración
a, b = 1, 4
n_values = [5, 10, 15]  # Subintervalos a evaluar

# Solución "exacta" usando scipy.integrate.quad
exact_integral, _ = quad(f, a, b)

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx, x_vals, y_vals = trapezoidal_rule(a, b, n)
    error = np.abs(exact_integral - approx)
    results.append((n, approx, error))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor de referencia (quad): {exact_integral:.8f}\n")
for n, approx, error in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.8f}")
    print(f"Error absoluto = {error:.4e}")
    print("---------------------------------")

# Gráfica de convergencia del error
n_list = [n for n, _, _ in results]
error_list = [err for _, _, err in results]

plt.figure(figsize=(8, 5))
plt.plot(n_list, error_list, 'bo-', label="Error absoluto")
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error absoluto")
plt.title("Convergencia del error en la regla del trapecio")
plt.yscale("log")
plt.xscale("log")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("convergencia_error.png")
plt.show()