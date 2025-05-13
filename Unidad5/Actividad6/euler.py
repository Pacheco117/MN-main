import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros del circuito RC
R = 1000  # Ohms
C = 0.001  # Faradios
V_fuente = 5  # Voltios

# Definición de la EDO: dV/dt = f(t, V)
def f(t, V):
    return (V_fuente - V) / (R * C)

# Condiciones iniciales
t0 = 0.0
V0 = 0.0
tf = 5.0
n = 20

# Paso
h = (tf - t0) / n

# Inicialización de listas para almacenar resultados
t_vals = [t0]
V_euler = [V0]

# Método de Euler
t = t0
V = V0
for _ in range(n):
    V = V + h * f(t, V)
    t = t + h
    t_vals.append(t)
    V_euler.append(V)

# Solución analítica
V_analitica = [V_fuente * (1 - np.exp(-t)) for t in t_vals]

# Cálculo de errores absolutos
error_absoluto = [abs(analit - euler) for analit, euler in zip(V_analitica, V_euler)]

# Crear DataFrame con los resultados
df = pd.DataFrame({
    't': t_vals,
    'V_euler': V_euler,
    'V_analitica': V_analitica,
    'error_absoluto': error_absoluto
})

# Guardar en CSV
csv_path = "euler_resultados_rc.csv"
df.to_csv(csv_path, index=False)

# Graficar ambas soluciones
plt.figure(figsize=(10, 6))
plt.plot(t_vals, V_euler, 'o-', label='Aproximación Euler', color='blue')
plt.plot(t_vals, V_analitica, '-', label='Solución Analítica', color='red')
plt.title('Carga de un Capacitor en Circuito RC')
plt.xlabel('Tiempo (s)')
plt.ylabel('Voltaje (V)')
plt.grid(True)
plt.legend()
plt.savefig("comparacion_rc.png")
plt.show()

# Análisis de desempeño
max_error = max(error_absoluto)
avg_error = np.mean(error_absoluto)
print(f"Máximo error absoluto: {max_error:.4f} V")
print(f"Error promedio absoluto: {avg_error:.4f} V")