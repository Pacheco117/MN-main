import numpy as np
import matplotlib.pyplot as plt

# Definición del sistema de EDOs: dy/dt = f(t, y)
def f(t, y):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -2 * y2 - 5 * y1
    return np.array([dy1dt, dy2dt])

# Método de Runge-Kutta de cuarto orden para sistemas
def runge_kutta_4_system(f, t0, y0, t_end, h):
    t_vals = [t0]
    y_vals = [y0.copy()]
    
    t = t0
    y = y0.copy()
    
    print(f"{'t (s)':>10} {'y1 (m)':>15} {'y2 (m/s)':>15}")
    print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")
    
    while t < t_end:
        k1 = f(t, y)
        k2 = f(t + h/2, y + (h/2) * k1)
        k3 = f(t + h/2, y + (h/2) * k2)
        k4 = f(t + h, y + h * k3)
        
        y += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        
        t_vals.append(t)
        y_vals.append(y.copy())
        
        print(f"{t:10.4f} {y[0]:15.6f} {y[1]:15.6f}")
    
    return t_vals, np.array(y_vals)

# Parámetros iniciales
t0 = 0
y0 = np.array([1.0, 0.0])  # y1(0) = 1, y2(0) = 0
t_end = 5
h = 0.1

# Llamada al método de Runge-Kutta
t_vals, y_vals = runge_kutta_4_system(f, t0, y0, t_end, h)

# Extraer y1 (posición) y y2 (velocidad)
y1_vals = y_vals[:, 0]
y2_vals = y_vals[:, 1]

# Solución exacta para comparación
def exact_solution(t):
    return np.exp(-t) * (np.cos(2*t) + 0.5 * np.sin(2*t))

y_exact = [exact_solution(t) for t in t_vals]

# Gráfico de la trayectoria
plt.figure(figsize=(10, 6))
plt.plot(t_vals, y1_vals, 'bo-', label="RK4 (h=0.1)", markersize=4)
plt.plot(t_vals, y_exact, 'r--', label="Solución Exacta", linewidth=1.5)
plt.xlabel("Tiempo (s)")
plt.ylabel("Posición (m)")
plt.title("Dinámica de un Resorte Amortiguado")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Análisis de errores (opcional)
max_error = np.max(np.abs(y1_vals - y_exact))
rms_error = np.sqrt(np.mean((y1_vals - y_exact)**2))

print("\nAnálisis de Desempeño:")
print(f"Error máximo absoluto: {max_error:.6f} m")
print(f"Error cuadrático medio (RMS): {rms_error:.6f} m")

# Gráfico del espacio de fases (opcional)
plt.figure(figsize=(8, 6))
plt.plot(y1_vals, y2_vals, 'b-', label="Trayectoria en el espacio de fases")
plt.xlabel("Posición (m)")
plt.ylabel("Velocidad (m/s)")
plt.title("Espacio de Fases del Sistema")
plt.grid(True)
plt.legend()
plt.show()