import numpy as np
import matplotlib.pyplot as plt

# Función a integrar: f(x) = x² + 3x + 1
def f(x):
    return x**2 + 3*x + 1

# Implementación de la regla del trapecio compuesta
def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral, x, y

# Parámetros de integración
a, b = 0, 2
n_values = [10, 20, 50]  # Subintervalos a evaluar

# Solución exacta de la integral
exact_integral = (2**3)/3 + (3*2**2)/2 + 2  # 32/3 ≈ 10.6666667

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx, x_vals, y_vals = trapezoidal_rule(a, b, n)
    error = np.abs(exact_integral - approx)
    results.append((n, approx, error))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor exacto: {exact_integral:.6f}\n")
for n, approx, error in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.6f}")
    print(f"Error absoluto = {error:.6e}")
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