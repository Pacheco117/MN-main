def calcular_errores(x, y, valor_real):
    # Calcula la diferencia entre los dos valores dados
    diferencia = x - y
    
    # Calcula el error absoluto como la diferencia entre el valor real esperado y la diferencia obtenida
    error_abs = abs(valor_real - diferencia)
    
    # Calcula el error relativo dividiendo el error absoluto entre el valor real absoluto
    error_rel = error_abs / abs(valor_real)
    
    # Convierte el error relativo en porcentaje
    error_pct = error_rel * 100
    
    # Imprime los resultados
    print(f"Diferencia: {diferencia}")
    print(f"Error absoluto: {error_abs}")
    print(f"Error relativo: {error_rel}")
    print(f"Error porcentual: {error_pct}%")
    
    # Retorna el error absoluto y relativo
    return error_abs, error_rel

# Lista de valores para probar el cálculo de errores
valores = [
    (1.0000001, 1.0000000, 0.0000001),  # Caso 1: Diferencia esperada muy pequeña
    (1.000000000000001, 1.000000000000000, 0.000000000000001)  # Caso 2: Diferencia aún más pequeña
]

# Itera sobre cada conjunto de valores y calcula los errores
for x, y, real in valores:
    print(f"\nPara x={x}, y={y}:")
    calcular_errores(x, y, real)