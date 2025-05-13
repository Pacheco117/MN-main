import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parámetros térmicos
T_amb = 25  # Temperatura ambiente en °C
k = 0.07    # Coeficiente de enfriamiento en 1/minuto

# Definición de la EDO: dT/dt = f(t, T)
def f(t, T):
    return -k * (T - T_amb)

# Condiciones iniciales
t0 = 0.0
T0 = 90.0
tf = 30.0  # Intervalo en minutos
n = 30

# Paso temporal
h = (tf - t0) / n

# Inicialización de listas para almacenar resultados
t_vals = [t0]
T_euler = [T0]

# Método de Euler
t = t0
T = T0
for _ in range(n):
    T = T + h * f(t, T)
    t = t + h
    t_vals.append(t)
    T_euler.append(T)

# Solución analítica
def temperatura_analitica(t):
    return T_amb + (T0 - T_amb) * np.exp(-k * t)

T_analitica = [temperatura_analitica(t) for t in t_vals]

# Cálculo de errores absolutos
error_absoluto = [abs(analit - euler) for analit, euler in zip(T_analitica, T_euler)]

# Crear DataFrame con resultados
df = pd.DataFrame({
    't (min)': t_vals,
    'T_euler (°C)': T_euler,
    'T_analitica (°C)': T_analitica,
    'error_absoluto (°C)': error_absoluto
})

# Guardar en CSV
csv_path = "euler_resultados_enfriamiento.csv"
df.to_csv(csv_path, index=False)

# Graficar ambas soluciones
plt.figure(figsize=(10, 6))
plt.plot(t_vals, T_euler, 'o-', label='Aproximación Euler', markersize=5, color='blue')
plt.plot(t_vals, T_analitica, '-', label='Solución Analítica', color='red')
plt.title('Enfriamiento de un Cuerpo (Ley de Newton)')
plt.xlabel('Tiempo (minutos)')
plt.ylabel('Temperatura (°C)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig("comparacion_enfriamiento.png")
plt.show()

# Análisis de desempeño
max_error = max(error_absoluto)
avg_error = np.mean(error_absoluto)
print("Análisis de desempeño:")
print(f"Máximo error absoluto: {max_error:.4f} °C")
print(f"Error promedio absoluto: {avg_error:.4f} °C")