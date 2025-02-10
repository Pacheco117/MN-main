import numpy as np
import matplotlib.pyplot as plt

epsilon = 1.0
iteracion = 0
iteraciones = []
precisiones = []

while 1.0 + epsilon != 1.0:
    epsilon /= 2
    iteracion += 1
    iteraciones.append(iteracion)
    precisiones.append(epsilon)
    print(f"Iteracion: {iteracion}, Precisión de máquina: {epsilon}")

epsilon *= 2
print(f"Precisión de máquina final: {epsilon}")

# Graficar los resultados
plt.plot(iteraciones, precisiones, marker='o')
plt.xlabel('Iteración')
plt.ylabel('Precisión de máquina (epsilon)')
plt.title('Convergencia de la precisión de máquina')
plt.grid(True)
plt.show()