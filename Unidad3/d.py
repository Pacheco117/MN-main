def es_diagonalmente_dominante(matriz):
    """
    Determina si una matriz es diagonalmente dominante.

    :param matriz: Una lista de listas que representa la matriz de coeficientes.
    :return: True si la matriz es diagonalmente dominante, False en caso contrario.
    """
    n = len(matriz)
    for i in range(n):
        # Suma de los valores absolutos de los elementos en la fila, excluyendo el elemento diagonal
        suma_fila = sum(abs(matriz[i][j]) for j in range(n) if j != i)
        # Compara el valor absoluto del elemento diagonal con la suma de los demás elementos
        if abs(matriz[i][i]) <= suma_fila:
            return False
    return True

# Ejemplo de uso
matriz = [
    [4, -1, 0],
    [1, 5, 2],
    [1, 1, 10]
]

if es_diagonalmente_dominante(matriz):
    print("La matriz es diagonalmente dominante.")
else:
    print("La matriz NO es diagonalmente dominante.")