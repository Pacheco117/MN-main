import numpy as np
import matplotlib.pyplot as plt

# Función a integrar: f(x) = sin(x)
def f(x):
    return np.sin(x)

# Implementación de la regla del trapecio compuesta
def trapezoidal_rule(a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    integral = (h / 2) * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return integral, x, y

# Parámetros de integración
a, b = 0, np.pi
n_values = [2, 4, 8, 16]  # Subintervalos a evaluar

# Solución exacta de la integral (∫₀^π sin(x) dx = 2)
exact_integral = 2.0

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx, x_vals, y_vals = trapezoidal_rule(a, b, n)
    error = np.abs(exact_integral - approx)
    results.append((n, approx, error, x_vals, y_vals))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor exacto: {exact_integral:.6f}\n")
for n, approx, error, _, _ in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.6f}")
    print(f"Error absoluto = {error:.4e}")
    print("---------------------------------")

# Gráfica de la función y aproximaciones para diferentes n
plt.figure(figsize=(12, 8))
x_fine = np.linspace(a, b, 100)
y_fine = f(x_fine)

# Subplots para cada n
for i, (n, _, _, x_vals, y_vals) in enumerate(results):
    plt.subplot(2, 2, i+1)
    plt.plot(x_fine, y_fine, 'r-', label='f(x) = sin(x)', linewidth=2)
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    plt.plot(x_vals, y_vals, 'bo-', label=f"n = {n}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Aproximación con n = {n}")
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.savefig("trapecio_sin.png")
plt.show()

# Gráfica de convergencia del error
n_list = [n for n, _, _, _, _ in results]
error_list = [err for _, _, err, _, _ in results]

plt.figure(figsize=(8, 5))
plt.plot(n_list, error_list, 'bo-', label="Error absoluto")
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error absoluto")
plt.title("Convergencia del error en la regla del trapecio")
plt.yscale("log")
plt.xscale("log")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("convergencia_error_sin.png")
plt.show()