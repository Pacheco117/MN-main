import numpy as np
import matplotlib.pyplot as plt

# Definir el sistema de ecuaciones actualizado (6x6)
A = np.array([
    [12, -2, 1, 0, 0, 0],
    [-3, 18, -4, 2, 0, 0],
    [1, -2, 16, -1, 1, 0],
    [0, 2, -1, 11, -3, 1],
    [0, 0, -2, 4, 15, -2],
    [0, 0, 0, 1, -3, 15]
])

b = np.array([20, 35, -5, 19, -12, 25])

# Solución exacta usando numpy.linalg.solve
sol_exacta = np.linalg.solve(A, b)

# Criterio de paro
tolerancia = 1e-6
max_iter = 100

# Implementación del método de Jacobi
def jacobi(A, b, tol, max_iter):
    n = len(A)
    x = np.zeros(n)  # Aproximación inicial
    errores_abs = []
    errores_rel = []
    errores_cuad = []
    
    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - suma) / A[i, i]
        
        # Calcular errores respecto a la solución exacta
        error_abs = np.linalg.norm(x_new - sol_exacta, ord=1)
        error_rel = np.linalg.norm(x_new - sol_exacta, ord=1) / np.linalg.norm(sol_exacta, ord=1)
        error_cuad = np.linalg.norm(x_new - sol_exacta, ord=2)
        
        errores_abs.append(error_abs)
        errores_rel.append(error_rel)
        errores_cuad.append(error_cuad)
        
        print(f"Iteración {k+1}: Error absoluto = {error_abs:.6f}, Error relativo = {error_rel:.6f}, Error cuadrático = {error_cuad:.6f}")
        
        # Verificar convergencia
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        
        x = x_new.copy()
    
    return x, errores_abs, errores_rel, errores_cuad, k+1

# Ejecutar el método
sol_aprox, errores_abs, errores_rel, errores_cuad, iteraciones = jacobi(A, b, tolerancia, max_iter)

# Graficar errores
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteraciones+1), errores_abs, label="Error absoluto", marker='o')
plt.plot(range(1, iteraciones+1), errores_rel, label="Error relativo", marker='s')
plt.plot(range(1, iteraciones+1), errores_cuad, label="Error cuadrático", marker='^')
plt.xlabel("Iteraciones")
plt.ylabel("Magnitud del Error")
plt.yscale("log")
plt.title("Convergencia de Errores - Método de Jacobi")
plt.legend()
plt.grid(True)
plt.savefig("errores_jacobi.png")
plt.show()

# Resultados
print("\nSolución aproximada:", sol_aprox)
print("Solución exacta:", sol_exacta)