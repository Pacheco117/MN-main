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
C = 1e-6  # Capacitancia en Faradios
T = 5.0   # Tiempo final en segundos

# Función de voltaje V(t) = 100e^(-2t)
def V(t):
    return 100 * np.exp(-2 * t)

# Solución analítica de la integral ∫₀^T V(t) dt
exact_integral = 50 * (1 - np.exp(-10))  # Integral exacta de V(t)
exact_charge = C * exact_integral        # Carga exacta Q = C * integral

# Subintervalos a evaluar
n_values = [6, 10, 20, 30]

# Calcular aproximaciones y errores para cada n
results = []
for n in n_values:
    approx_integral = simpson_rule(V, 0, T, n)
    approx_charge = C * approx_integral
    error = np.abs(exact_charge - approx_charge)
    results.append((n, approx_charge, error))

# Imprimir resultados
print("Resultados de la aproximación:")
print("---------------------------------")
print(f"Valor exacto: {exact_charge:.16f} C\n")
for n, approx, error in results:
    print(f"n = {n}:")
    print(f"Aproximación = {approx:.16f} C")
    print(f"Error absoluto = {error:.4e} C")
    print("---------------------------------")

# Gráfica de convergencia del error
n_list = [n for n, _, _ in results]
error_list = [err for _, _, err in results]

plt.figure(figsize=(8, 5))
plt.plot(n_list, error_list, 'bo-', label="Error absoluto")
plt.xlabel("Número de subintervalos (n)")
plt.ylabel("Error absoluto (C)")
plt.title("Convergencia del error en la regla de Simpson")
plt.yscale("log")
plt.xscale("log")
plt.grid(which="both", linestyle="--")
plt.legend()
plt.savefig("convergencia_error_capacitor.png")
plt.show()

# Gráfica de la función y puntos de interpolación (ejemplo para n=10)
n_demo = 10
x_vals = np.linspace(0, T, n_demo + 1)
y_vals = V(x_vals)
x_fine = np.linspace(0, T, 100)
y_fine = V(x_fine)

plt.figure(figsize=(8, 5))
plt.plot(x_fine, y_fine, 'b-', label=r'$V(t) = 100e^{-2t}$', linewidth=2)
plt.fill_between(x_fine, y_fine, alpha=0.3, color="cyan", label="Área aproximada")
plt.scatter(x_vals, y_vals, color="red", label=f"Puntos de interpolación (n={n_demo})")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.legend()
plt.title("Aproximación de la integral con la regla de Simpson")
plt.grid()
plt.savefig("simpson_capacitor.png")
plt.show()