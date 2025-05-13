import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del sistema
g = 9.81  # m/s²
m = 2.0   # kg
k = 0.5   # kg/s

# Definición de la EDO: dv/dt = f(t, v)
def f(t, v):
    return g - (k/m) * v

# Condiciones iniciales
t0 = 0.0
v0 = 0.0
tf = 10.0
n = 50

# Paso temporal
h = (tf - t0) / n

# Inicialización de listas para almacenar resultados
t_vals = [t0]
v_euler = [v0]

# Método de Euler
t = t0
v = v0
for _ in range(n):
    v = v + h * f(t, v)
    t = t + h
    t_vals.append(t)
    v_euler.append(v)

# Solución analítica
def velocidad_analitica(t):
    return (m * g / k) * (1 - np.exp(-(k/m) * t))

v_analitica = [velocidad_analitica(t) for t in t_vals]

# Cálculo de errores absolutos
error_absoluto = [abs(analit - euler) for analit, euler in zip(v_analitica, v_euler)]

# Crear DataFrame con resultados
df = pd.DataFrame({
    't': t_vals,
    'v_euler': v_euler,
    'v_analitica': v_analitica,
    'error_absoluto': error_absoluto
})

# Guardar en CSV
csv_path = "euler_resultados_caida.csv"
df.to_csv(csv_path, index=False)

# Graficar ambas soluciones
plt.figure(figsize=(10, 6))
plt.plot(t_vals, v_euler, 'o-', label='Aproximación Euler', markersize=4, color='blue')
plt.plot(t_vals, v_analitica, '-', label='Solución Analítica', color='red')
plt.title('Caída Libre con Resistencia del Aire')
plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad (m/s)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("comparacion_caida.png")
plt.show()

