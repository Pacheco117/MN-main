import numpy as np
import matplotlib.pyplot as plt
import csv

# Definir el sistema actualizado (4x4)
A = np.array([
    [10, 2, 3, 1],
    [2, 12, 2, 3],
    [3, 2, 15, 1],
    [1, 3, 1, 10]
])

b = np.array([15, 22, 18, 10])

# Solución exacta usando numpy.linalg.solve
sol_exacta = np.linalg.solve(A, b)

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)  # Inicialización en 0
    x_prev = np.copy(x)
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    
    for k in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        # Calcular errores respecto a la solución exacta
        error_abs = np.linalg.norm(x - sol_exacta, ord=1)
        error_rel = error_abs / (np.linalg.norm(sol_exacta, ord=1) + 1e-10)  # Evitar división por cero
        error_cuad = np.linalg.norm(x - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        # Criterio de paro basado en la diferencia entre iteraciones
        if np.linalg.norm(x - x_prev, ord=np.inf) < tol:
            break
        
        x_prev = np.copy(x)
    
    # Preparar lista de errores para CSV (iteración, abs, rel, cuad)
    errors = list(zip(range(1, len(errores_abs)+1), errores_abs, errores_rel, errores_cuad))
    return x, errors

# Ejecutar Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)

# Guardar errores en CSV
with open("errores_gauss_seidel.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Iteración", "Error Absoluto", "Error Relativo", "Error Cuadrático"])
    writer.writerows(errors)
    writer.writerow([])
    writer.writerow(["Solución aproximada"])
    writer.writerow(x_sol)

# Graficar errores
iteraciones = [e[0] for e in errors]
abs_errors = [e[1] for e in errors]
rel_errors = [e[2] for e in errors]
cuad_errors = [e[3] for e in errors]

plt.figure(figsize=(10, 6))
plt.plot(iteraciones, abs_errors, label="Error Absoluto", marker='o')
plt.plot(iteraciones, rel_errors, label="Error Relativo", marker='s')
plt.plot(iteraciones, cuad_errors, label="Error Cuadrático", marker='^')
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Magnitud del Error")
plt.title("Convergencia del Método de Gauss-Seidel")
plt.legend()
plt.grid(True)
plt.savefig("convergencia_gauss_seidel.png")
plt.show()

# Mostrar resultados
print("Solución aproximada:", x_sol)
print("Solución exacta:", sol_exacta)