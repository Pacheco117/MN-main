import numpy as np
import matplotlib.pyplot as plt

# Función a evaluar
def f(x):
    return np.sin(x)

# Derivada analítica
def df_analytical(x):
    return np.cos(x)

# Métodos de diferencias finitas
def forward_diff(f, x, h=0.1):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h=0.1):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h=0.1):
    return (f(x + h) - f(x - h)) / (2 * h)

# Intervalo y paso
a = 0.0
b = np.pi
h = 0.1

# Puntos de evaluación con paso h
x_eval = np.arange(a, b + h, h)
x_eval = x_eval[x_eval <= b]  # Asegurar que no exceda π

# Puntos densos para graficar
x_plot = np.linspace(a, b, 100)

# Derivada exacta en puntos de graficado
df_exact_plot = df_analytical(x_plot)

# Aproximaciones numéricas en puntos de evaluación
df_forward = forward_diff(f, x_eval, h)
df_backward = backward_diff(f, x_eval, h)
df_central = central_diff(f, x_eval, h)

# Cálculo de errores absolutos
df_exact_eval = df_analytical(x_eval)
error_forward = np.abs(df_forward - df_exact_eval)
error_backward = np.abs(df_backward - df_exact_eval)
error_central = np.abs(df_central - df_exact_eval)

# Gráfica de comparación
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), '-', label='Función')
plt.plot(x_plot, df_exact_plot, 'k-', label='Derivada Analítica')
plt.plot(x_eval, df_forward, 'r--', label='Hacia adelante')
plt.plot(x_eval, df_backward, 'g-.', label='Hacia atrás')
plt.plot(x_eval, df_central, 'b:', label='Centrada')
plt.xlabel('x')
plt.ylabel("Derivada")
plt.legend()
plt.title("Comparación de Métodos de Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_aproximaciones.png")
plt.show()

# Gráfica de errores
plt.figure(figsize=(10, 6))
plt.plot(x_eval, error_forward, 'r--', label='Error Hacia adelante')
plt.plot(x_eval, error_backward, 'g-.', label='Error Hacia atrás')
plt.plot(x_eval, error_central, 'b:', label='Error Centrada')
plt.xlabel('x')
plt.ylabel("Error absoluto")
plt.legend()
plt.title("Errores en Diferenciación Numérica")
plt.grid()
plt.savefig("diferenciacion_errores.png")
plt.show()