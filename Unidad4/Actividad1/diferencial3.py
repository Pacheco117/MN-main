import numpy as np
import matplotlib.pyplot as plt

# Función a evaluar: f(x) = x^3 - 2x² + x
def f(x):
    return x**3 - 2*x**2 + x

# Derivada analítica: f'(x) = 3x² - 4x + 1
def df_analytical(x):
    return 3*x**2 - 4*x + 1

# Métodos de diferencias finitas (h=0.2 por defecto)
def forward_diff(f, x, h=0.2):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=0.2):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=0.2):
    return (f(x + h) - f(x - h)) / (2 * h)

# Intervalo y paso
a = -1.0
b = 2.0
h = 0.2

# Puntos de evaluación con paso h (asegurando límites)
x_eval = np.arange(a, b + h, h)
x_eval = x_eval[x_eval <= b]  # Eliminar puntos fuera del intervalo

# Puntos densos para graficar
x_plot = np.linspace(a, b, 100)

# Aproximaciones numéricas
df_forward = forward_diff(f, x_eval, h)
df_backward = backward_diff(f, x_eval, h)
df_central = central_diff(f, x_eval, h)

# Derivada exacta en puntos de evaluación y graficado
df_exact_eval = df_analytical(x_eval)
df_exact_plot = df_analytical(x_plot)

# Cálculo de errores absolutos
error_forward = np.abs(df_forward - df_exact_eval)
error_backward = np.abs(df_backward - df_exact_eval)
error_central = np.abs(df_central - df_exact_eval)

# Gráfica de comparación
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), '-', label='Función')
plt.plot(x_plot, df_exact_plot, 'k-', label='Derivada Analítica')
plt.plot(x_eval, df_forward, 'r--', label='Hacia adelante (h=0.2)')
plt.plot(x_eval, df_backward, 'g-.', label='Hacia atrás (h=0.2)')
plt.plot(x_eval, df_central, 'b:', label='Centrada (h=0.2)')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica para f(x) = x³ - 2x² + x")
plt.grid()
plt.savefig("derivadas_polinomio.png")
plt.show()

# Gráfica de errores
plt.figure(figsize=(10, 6))
plt.plot(x_eval, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_eval, error_backward, 'g-.', label='Error Hacia atrás')
plt.plot(x_eval, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.yscale('log')  # Escala logarítmica para mejor visualización
plt.legend()
plt.title("Errores Absolutos en Diferenciación Numérica (h=0.2)")
plt.grid()
plt.savefig("errores_polinomio.png")
plt.show()