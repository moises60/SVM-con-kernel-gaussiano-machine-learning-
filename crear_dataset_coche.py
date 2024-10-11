

import numpy as np
import pandas as pd

# Establecer la semilla para reproducibilidad
np.random.seed(42)

# Número de muestras
n_samples = 1500

# Generar edades entre 20 y 70 años
edad = np.random.randint(20, 71, n_samples)

# Generar poder económico (0: Bajo, 1: Medio, 2: Alto)
poder_economico = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])

# Crear un DataFrame
datos = pd.DataFrame({
    'Edad': edad,
    'Poder_Economico': poder_economico
})

# Función para determinar si compró el coche (1) o no (0)
def comprar_coche(row):
    probabilidad = 0.0
    # Aumentar la probabilidad según el poder económico
    if row['Poder_Economico'] == 2:  # Alto
        probabilidad += 0.7
    elif row['Poder_Economico'] == 1:  # Medio
        probabilidad += 0.33
    else:  # Bajo
        probabilidad += 0.09

    # Aumentar la probabilidad según la edad
    if 30 <= row['Edad'] <= 50:
        probabilidad += 0.3
    elif 51 <= row['Edad'] <= 65:
        probabilidad += 0.2
    else:
        probabilidad += 0.1

    # Limitar la probabilidad máxima a 0.99
    probabilidad = min(probabilidad, 0.99)

    # Determinar si compra o no
    return 1 if np.random.rand() < probabilidad else 0

# Aplicar la función al DataFrame
datos['Compra'] = datos.apply(comprar_coche, axis=1)

# Mapear valores para mejor legibilidad
datos['Poder_Economico'] = datos['Poder_Economico'].map({0: 'Bajo', 1: 'Medio', 2: 'Alto'})

# Guardar el dataset a un archivo CSV
datos.to_csv('datos_coche.csv', index=False)

print("Dataset 'datos_coche.csv' creado exitosamente.")
