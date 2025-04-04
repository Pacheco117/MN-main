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

# Parámetros del problema
k = 0.5  # Conductividad térmica en W/m-K
x1, x2 = 0.0, 2.0  # Intervalo de integración

# Función a integrar: dT/dx = -100x
def dT_dx(x):
    return -100 * x

# Solución analítica de la integral ∫₀² (-100x) dx = -200
exact_integral = -200.0
exact_heat_flux = k * exact_integral  # Q = -100 W

# Subintervalos a evaluar
n_values = [6, 10, 20, 30]

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx_integral = simpson_rule(dT_dx, x1, x2, n)
    approx_heat_flux = k * approx_integral
    error = np.abs(exact_heat_flux - approx_heat_flux)
    results.append((n, approx_heat_flux, error))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor exacto: Q = {exact_heat_flux:.6f} W\n")
for n, approx, error in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.16f} W")
    print(f"Error absoluto = {error:.2e} W")
    print("---------------------------------")

# Gráfica de convergencia del error
n_list = [n for n, _, _ in results]
error_list = [err for _, _, err in results]

plt.figure(figsize=(8, 5))
plt.plot(n_list, error_list, 'bo-', label="Error absoluto")
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error absoluto (W)")
plt.title("Convergencia del error en la regla de Simpson")
plt.yscale("log")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("convergencia_error_calor.png")
plt.show()

# Gráfica de la función y puntos de interpolación (ejemplo para n=10)
n_demo = 10
x_vals = np.linspace(x1, x2, n_demo + 1)
y_vals = dT_dx(x_vals)
x_fine = np.linspace(x1, x2, 100)
y_fine = dT_dx(x_fine)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, y_fine, 'b-', label=r'$\frac{dT}{dx} = -100x$', linewidth=2)
plt.fill_between(x_fine, y_fine, alpha=0.3, color="cyan", label="Área aproximada")
plt.scatter(x_vals, y_vals, color="red", label=f"Puntos de interpolación (n={n_demo})")
plt.xlabel("Posición (m)")
plt.ylabel("Gradiente de temperatura (K/m)")
plt.legend()
plt.title("Aproximación del flujo de calor con la regla de Simpson")
plt.grid()
plt.savefig("simpson_calor.png")
plt.show()