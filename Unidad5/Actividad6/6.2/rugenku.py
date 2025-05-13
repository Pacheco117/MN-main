import numpy as np
import matplotlib.pyplot as plt

# Definición de la EDO: dT/dx = f(x, T)
def f(x, T):
    return -0.25 * (T - 25)

# Método de Runge-Kutta de cuarto orden
def runge_kutta_4(f, x0, T0, x_end, h):
    x_vals = [x0]
    T_vals = [T0]
    
    x = x0
    T = T0
    
    print(f"{'x':>10} {'T (RK4)':>15}")
    print(f"{x:10.4f} {T:15.6f}")
    
    while x < x_end:
        k1 = f(x, T)
        k2 = f(x + h/2, T + h/2 * k1)
        k3 = f(x + h/2, T + h/2 * k2)
        k4 = f(x + h, T + h * k3)
        
        T += h * (k1 + 2*k2 + 2*k3 + k4) / 6
        x += h
        
        x_vals.append(x)
        T_vals.append(T)
        
        print(f"{x:10.4f} {T:15.6f}")
    
    return x_vals, T_vals

# Parámetros iniciales
x0 = 0
T0 = 100
x_end = 2
h = 0.1

# Llamada al método de Runge-Kutta
x_vals, T_vals_rk4 = runge_kutta_4(f, x0, T0, x_end, h)

# Solución exacta
T_exact_vals = [25 + 75 * np.exp(-0.25 * x) for x in x_vals]

# Tabla de comparación
print("\nComparación con la solución exacta:")
print(f"{'x':>10} {'T (RK4)':>15} {'T (Exacta)':>15} {'Error':>15}")
for x, Trk4, Texact in zip(x_vals, T_vals_rk4, T_exact_vals):
    error = abs(Texact - Trk4)
    print(f"{x:10.4f} {Trk4:15.6f} {Texact:15.6f} {error:15.6f}")

# Cálculo de errores
max_error = max(abs(Texact - Trk4) for Texact, Trk4 in zip(T_exact_vals, T_vals_rk4))
rms_error = np.sqrt(np.mean([(Texact - Trk4)**2 for Texact, Trk4 in zip(T_exact_vals, T_vals_rk4)]))

print("\nAnálisis de desempeño:")
print(f"Máximo error absoluto: {max_error:.6f}")
print(f"Error cuadrático medio (RMS): {rms_error:.6f}")

# Graficar ambas soluciones
plt.figure(figsize=(8,5))
plt.plot(x_vals, T_vals_rk4, 'bo-', label="RK4 (h=0.1)")
plt.plot(x_vals, T_exact_vals, 'r--', label="Solución Exacta")
plt.xlabel("x (metros)")
plt.ylabel("Temperatura (°C)")
plt.title("Perfil de Temperatura en el Tubo")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()