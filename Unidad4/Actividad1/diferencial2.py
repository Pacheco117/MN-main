import numpy as np
import matplotlib.pyplot as plt

# Función a evaluar: f(x) = e^x
def f(x):
    return np.exp(x)

# Derivada analítica: f'(x) = e^x
def df_analytical(x):
    return np.exp(x)

# Métodos de diferencias finitas (h=0.05 por defecto)
def forward_diff(f, x, h=0.05):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=0.05):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=0.05):
    return (f(x + h) - f(x - h)) / (2 * h)

# Intervalo y paso
a = 0.0
b = 2.0
h = 0.05

# Puntos de evaluación con paso h
x_eval = np.arange(a, b + h, h)
x_eval = x_eval[x_eval <= b]  # Asegurar que no exceda b

# Puntos densos para graficar la función y derivada exacta
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
plt.plot(x_plot, f(x_plot), '-', label='Función (e^x)')
plt.plot(x_plot, df_exact_plot, 'k-', label='Derivada Analítica (e^x)')
plt.plot(x_eval, df_forward, 'r--', label='Hacia adelante (h=0.05)')
plt.plot(x_eval, df_backward, 'g-.', label='Hacia atrás (h=0.05)')
plt.plot(x_eval, df_central, 'b:', label='Centrada (h=0.05)')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica para f(x) = e^x")
plt.grid()
plt.savefig("derivadas_exponencial.png")
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
plt.title("Errores Absolutos en Diferenciación Numérica (h=0.05)")
plt.grid()
plt.savefig("errores_exponencial.png")
plt.show()