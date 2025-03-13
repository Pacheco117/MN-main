import numpy as np
import matplotlib.pyplot as plt
import csv

# Definir el sistema 10x10
A = np.array([
    [15, -4, -1, -2, 0, 0, 0, 0, 0, 0],    # Ecuación 1
    [-3, 18, -2, 0, -1, 0, 0, 0, 0, 0],     # Ecuación 2
    [-1, -2, 20, 0, 0, -5, 0, 0, 0, 0],     # Ecuación 3
    [-2, -1, -4, 22, 0, 0, -1, 0, 0, 0],    # Ecuación 4
    [0, -1, -3, -1, 25, 0, 0, -2, 0, 0],    # Ecuación 5
    [0, 0, -2, 0, -1, 28, 0, 0, -1, 0],     # Ecuación 6
    [0, 0, 0, -4, 0, -2, 30, 0, 0, -3],     # Ecuación 7
    [0, 0, 0, 0, -1, 0, -1, 35, -2, 0],     # Ecuación 8
    [0, 0, 0, 0, 0, -2, 0, -3, 40, -1],     # Ecuación 9
    [0, 0, 0, 0, 0, 0, -3, 0, -1, 45]       # Ecuación 10
])

b = np.array([200, 250, 180, 300, 270, 310, 320, 400, 450, 500])

# Solución exacta
sol_exacta = np.linalg.solve(A, b)

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    x_prev = np.copy(x)
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    
    for k in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        error_abs = np.linalg.norm(x - sol_exacta, ord=1)
        error_rel = error_abs / (np.linalg.norm(sol_exacta, ord=1) + 1e-10)
        error_cuad = np.linalg.norm(x - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break
        x_prev = np.copy(x)
    
    # Corregir la línea de errores usando zip correctamente
    errors = list(zip(range(1, len(errores_abs)+1), errores_abs, errores_rel, errores_cuad))
    return x, errors

# Ejecutar método
x_sol, errors = gauss_seidel(A, b)

# Guardar resultados
with open("errores_gauss_seidel.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteración", "Error Absoluto", "Error Relativo", "Error Cuadrático"])
    writer.writerows(errors)
    writer.writerow(["Solución aproximada:", x_sol])

# Graficar
iteraciones = [e[0] for e in errors]
plt.figure(figsize=(10, 6))
plt.plot(iteraciones, [e[1] for e in errors], label="Error Absoluto", marker='o')
plt.plot(iteraciones, [e[2] for e in errors], label="Error Relativo", marker='s')
plt.plot(iteraciones, [e[3] for e in errors], label="Error Cuadrático", marker='^')
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.title("Convergencia del Método de Gauss-Seidel")
plt.legend()
plt.grid()
plt.savefig("convergencia.png")
plt.show()

print("Solución aproximada:", x_sol)
print("Solución exacta:", sol_exacta)