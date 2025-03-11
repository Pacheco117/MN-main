import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Eliminación hacia adelante
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Definición del sistema de ecuaciones
A = np.array([[6, -2, 3, -1, 2],
              [-3, 5, -2, 4, -1],
              [4, 3, 7, -5, 3],
              [-2, 6, -3, 1, -4],
              [1, -3, 2, -5, 6]], dtype=float)
b = np.array([15, -6, 20, -4, 7], dtype=float)

# Resolución del sistema
sol = gauss_elimination(A.copy(), b.copy())

# Impresión de resultados
print("Tabla de resultados:")
print("--------------------------------")
print("| Variable | Valor  |")
print("--------------------------------")
for i, val in enumerate(sol):
    print(f"|    x{i+1}    |  {val:.4f} |")
print("--------------------------------")

# Cálculo del error (norma del residuo)
residuo = np.dot(A, sol) - b
error = np.linalg.norm(residuo)
print(f"Norma del residuo (error): {error:.6f}")

# Gráfica del error de cada ecuación
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(residuo) + 1), np.abs(residuo), color='blue')
plt.xlabel('Ecuación')
plt.ylabel('Error absoluto')
plt.title('Error absoluto en cada ecuación')
plt.xticks(range(1, len(residuo) + 1))
plt.show()
