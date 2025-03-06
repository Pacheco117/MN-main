import numpy as np

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
# La matriz A se incluye por renglones
# El vector b se incluye por columnas
A = np.array([[2, 3, -1], [4, 1, 2], [-2, 5, 2]], dtype=float)
b = np.array([5, 6, -3], dtype=float)

# Resolución del sistema
sol = gauss_elimination(A, b)

# Imprimir la solución
print("Solución del sistema:")
print(sol)