import numpy as np
import matplotlib.pyplot as plt

# Función original f(x) = e^(-x) - x
def f(x):
    return np.exp(-x) - x

# Interpolación de Lagrange
def lagrange_interpolation(x, x_points, y_points):
    n = len(x_points)
    result = 0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

# Método de Bisección
def bisect(func, a, b, tol=1e-6, max_iter=100):
    if func(a) * func(b) > 0:
        raise ValueError("El intervalo no contiene una raíz")
    
    iter_count = 0
    errors = []
    while (b - a) / 2 > tol and iter_count < max_iter:
        c = (a + b) / 2
        errors.append(abs(func(c)))
        if abs(func(c)) < tol:
            return c, errors
        if func(a) * func(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1
    return (a + b) / 2, errors

# Selección de cuatro puntos equidistantes en [0,1]
x_points = np.linspace(0, 1, 4)
y_points = f(x_points)

# Construcción del polinomio interpolante
x_vals = np.linspace(0, 1, 100)
y_interp = [lagrange_interpolation(x, x_points, y_points) for x in x_vals]

# Encontrar raíz del polinomio interpolante usando bisección
root, errors = bisect(lambda x: lagrange_interpolation(x, x_points, y_points), 0, 1)

# Cálculo de errores
true_root = 0.56714329  # Aproximación conocida de la raíz real
total_error = abs(root - true_root)

# Gráfica de la función y el polinomio interpolante
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f(x_vals), label="f(x) = e^(-x) - x", linestyle='dashed', color='blue')
plt.plot(x_vals, y_interp, label="Polinomio Interpolante de Lagrange", color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(root, color='green', linestyle='dotted', label=f"Raíz aproximada: {root:.6f}")
plt.scatter(x_points, y_points, color='black', label="Puntos de interpolación")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Interpolación de Lagrange y Búsqueda de Raíces")
plt.legend()
plt.grid(True)
plt.savefig("interpolacion_raices.png")
plt.show()

# Gráfica de la convergencia del error
plt.figure(figsize=(8, 6))
plt.plot(range(len(errors)), errors, marker='o', linestyle='-', color='purple')
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Error absoluto")
plt.title("Convergencia del método de bisección")
plt.grid(True)
plt.savefig("error_convergencia.png")
plt.show()

# Imprimir resultados y errores
print(f"Raíz aproximada usando interpolación: {root:.6f}")
print(f"Error absoluto: {total_error:.6f}")
