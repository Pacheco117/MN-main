import numpy as np
import matplotlib.pyplot as plt

def gauss_jordan_pivot_determinante(A, b):
    """
    Resuelve un sistema de ecuaciones Ax = b mediante el método de Gauss-Jordan con pivoteo parcial
    e imprime el determinante de A para verificar si el sistema tiene solución única.
    """
    n = len(A)
    # Matriz aumentada
    Ab = np.hstack([A, b.reshape(-1, 1)]).astype(float)
    
    # Cálculo del determinante de A
    det_A = np.linalg.det(A)
    
    # Verificar si el sistema es determinado o indeterminado
    if np.isclose(det_A, 0):
        print(f"Determinante de A: {det_A:.5f}. El sistema es indeterminado o no tiene solución única.")
        return None
    
    print(f"Determinante de A: {det_A:.5f}. El sistema tiene solución única.")
    
    # Aplicación del método de Gauss-Jordan con pivoteo
    for i in range(n):
        # Pivoteo parcial
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            Ab[[i, max_row]] = Ab[[max_row, i]]

        # Normalización de la fila pivote
        Ab[i] = Ab[i] / Ab[i, i]

        # Eliminación en otras filas
        for j in range(n):
            if i != j:
                Ab[j] -= Ab[j, i] * Ab[i]

    # Extraer la solución
    x = Ab[:, -1]
    return x

# Definir el sistema de ecuaciones
A = np.array([[2, -3, 4, -1, 5, -1, 2, -1, 3, -2],
              [-3, 2, 5, 3, 4, 2, -3, -1, 2, 5],
              [4, -1, -2, 3, -3, 5, -2, 3, -4, 9],
              [-1, 5, -2, 3, 4, -5, -3, 2, 3, -10],
              [3, -2, 5, -3, 4, 2, -3, -5, 2, 1],
              [-2, 4, -2, -3, 3, 4, -3, 3, -2, -1],
              [5, -1, 2, -3, -4, 5, -2, -3, 1, 6],
              [1, -3, 2, 3, -4, -2, 5, -7, 8, 3],
              [2, -3, 4, -1, 5, -1, 2, -1, 3, -2],
              [-3, 2, 4, -3, -1, -2, 5, -7, 8, -4]])

b = np.array([11, -10, 8, 6, 7, -3, 9, 6, 5, -8])

# Resolver el sistema
solucion = gauss_jordan_pivot_determinante(A, b)

# Imprimir la solución si existe
if solucion is not None:
    print("Solución del sistema:")
    for i, val in enumerate(solucion):
        print(f"x{i+1} = {val:.5f}")
    
    # Gráfica de los valores de la solución
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(solucion) + 1), solucion, color='skyblue')
    plt.xlabel("Variable")
    plt.ylabel("Valor de la solución")
    plt.title("Solución del sistema de ecuaciones")
    plt.xticks(range(1, len(solucion) + 1), [f"x{i+1}" for i in range(len(solucion))])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    # Análisis del error
    error = np.abs(np.dot(A, solucion) - b)
    print("Errores absolutos por ecuación:")
    for i, err in enumerate(error):
        print(f"Ecuación {i+1}: {err:.5f}")
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, len(error) + 1), error, color='red')
    plt.xlabel("Ecuación")
    plt.ylabel("Error absoluto")
    plt.title("Análisis de error en la solución")
    plt.xticks(range(1, len(error) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
