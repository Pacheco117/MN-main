import numpy as np
import matplotlib.pyplot as plt

def simpson_rule(f, a, b, n):
    """Aproxima la integral de f(x) en [a, b] usando la regla de Simpson."""
    if n % 2 == 1:
        raise ValueError("El número de subintervalos (n) debe ser par.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    fx = f(x)
    integral = (h / 3) * (fx[0] + 2 * np.sum(fx[2:-1:2]) + 4 * np.sum(fx[1:-1:2]) + fx[-1])
    return integral

# Función a integrar: f(x) = kx (k = 200 N/m)
def funcion(x):
    return 200 * x

# Parámetros de integración
a, b = 0.1, 0.3  # Intervalo [a, b]
exact_integral = 8.0  # Solución analítica: (200/2)(0.3² - 0.1²) = 8
n_values = [6, 10, 20, 30]  # Subintervalos a evaluar

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx = simpson_rule(funcion, a, b, n)
    error = np.abs(exact_integral - approx)
    results.append((n, approx, error))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor exacto: {exact_integral}\n")
for n, approx, error in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.16f}")
    print(f"Error absoluto = {error:.2e}")
    print("---------------------------------")

# Gráfica de convergencia del error
n_list = [n for n, _, _ in results]
error_list = [err for _, _, err in results]

plt.figure(figsize=(8, 5))
plt.plot(n_list, error_list, 'bo-', label="Error absoluto")
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error absoluto")
plt.title("Convergencia del error en la regla de Simpson")
plt.yscale("log")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("convergencia_error.png")
plt.show()