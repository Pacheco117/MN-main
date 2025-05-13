import numpy as np
import matplotlib.pyplot as plt

# Definición de la EDO: dq/dt = f(t, q)
def f(t, q):
    V = 10     # Voltaje (V)
    R = 1000   # Resistencia (Ω)
    C = 0.001  # Capacitancia (F)
    return (V - q/C) / R

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, t0, q0, t_end, h):
    t_vals = [t0]
    q_vals = [q0]
    
    t = t0
    q = q0
    
    print(f"{'t (s)':>10} {'q (C) - RK4':>15}")
    print(f"{t:10.4f} {q:15.6f}")
    
    while t < t_end:
        k1 = f(t, q)
        k2 = f(t + h/2, q + h/2 * k1)
        k3 = f(t + h/2, q + h/2 * k2)
        k4 = f(t + h, q + h * k3)
        
        q += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += h
        
        t_vals.append(t)
        q_vals.append(q)
        
        print(f"{t:10.4f} {q:15.6f}")
    
    return t_vals, q_vals

# Parámetros iniciales
t0 = 0
q0 = 0
t_end = 1
h = 0.05

# Llamada al método de Runge-Kutta
t_vals, q_vals_rk4 = runge_kutta_4(f, t0, q0, t_end, h)

# Solución exacta
q_exact_vals = [0.01 * (1 - np.exp(-t)) for t in t_vals]

# Tabla de comparación
print("\nComparación con la solución exacta:")
print(f"{'t (s)':>10} {'q (C) - RK4':>15} {'q (C) - Exacta':>15} {'Error':>15}")
for t, q_rk4, q_exact in zip(t_vals, q_vals_rk4, q_exact_vals):
    error = abs(q_exact - q_rk4)
    print(f"{t:10.4f} {q_rk4:15.6f} {q_exact:15.6f} {error:15.6f}")

# Cálculo de errores
max_error = max(abs(q_exact - q_rk4) for q_exact, q_rk4 in zip(q_exact_vals, q_vals_rk4))
rms_error = np.sqrt(np.mean([(q_exact - q_rk4)**2 for q_exact, q_rk4 in zip(q_exact_vals, q_vals_rk4)]))

print("\nAnálisis de desempeño:")
print(f"Máximo error absoluto: {max_error:.6f} C")
print(f"Error cuadrático medio (RMS): {rms_error:.6f} C")

# Graficar ambas soluciones
plt.figure(figsize=(8,5))
plt.plot(t_vals, q_vals_rk4, 'bo-', label="RK4 (h=0.05)")
plt.plot(t_vals, q_exact_vals, 'r--', label="Solución Exacta")
plt.xlabel("Tiempo (s)")
plt.ylabel("Carga del capacitor (C)")
plt.title("Carga de un Capacitor en Circuito RC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()