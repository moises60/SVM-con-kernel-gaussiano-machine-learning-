

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Configurar estilo de los gráficos
sns.set(style='whitegrid')

# Cargar el dataset
datos = pd.read_csv('datos_coche.csv')

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(datos.head())

# Descripción estadística del dataset
print("\nDescripción estadística del dataset:")
print(datos.describe(include='all'))

# Verificar la distribución de la variable objetivo
print("\nDistribución de la variable 'Compra':")
print(datos['Compra'].value_counts())

# Visualizar la distribución de la variable 'Compra'
plt.figure(figsize=(6,4))
sns.countplot(x='Compra', data=datos)
plt.title('Distribución de la Variable Objetivo (Compra)')
plt.show()

# Preprocesamiento de Datos
# Convertir 'Poder_Economico' en variable numérica
datos['Poder_Economico_Num'] = datos['Poder_Economico'].map({'Bajo': 0, 'Medio': 1, 'Alto': 2})

# Seleccionar variables para el modelo
X = datos[['Edad', 'Poder_Economico_Num']]
y = datos['Compra']

# Dividir el dataset en conjunto de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Escalado de variables
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_entrenamiento = scaler.fit_transform(X_entrenamiento)
X_prueba = scaler.transform(X_prueba)

# Entrenar el modelo SVM con kernel Gaussiano (RBF)
from sklearn.svm import SVC

modelo = SVC(kernel='rbf', gamma='scale', C=1.0, random_state=42)
modelo.fit(X_entrenamiento, y_entrenamiento)

# Predicción de los resultados con el Conjunto de Prueba
y_pred = modelo.predict(X_prueba)

# Evaluación del modelo
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("\nMatriz de Confusión:")
print(confusion_matrix(y_prueba, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_prueba, y_pred))

print(f"Precisión del modelo: {accuracy_score(y_prueba, y_pred):.2f}")

# Visualización de la Matriz de Confusión
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_prueba, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Verdaderos')
plt.xlabel('Predicciones')
plt.show()

# Representación gráfica de los resultados en el Conjunto de Prueba
X_set, y_set = X_prueba, y_prueba.values
X1, X2 = np.meshgrid(
    np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
)

plt.figure(figsize=(10, 6))
plt.contourf(
    X1, X2,
    modelo.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75, cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Colores para los puntos
colors = ['red' if y == 0 else 'green' for y in y_set]

# Graficar los puntos de datos
plt.scatter(
    X_set[:, 0], X_set[:, 1],
    c=colors, edgecolor='k', s=20
)
plt.title('SVM con Kernel Gaussiano (Conjunto de Prueba)')
plt.xlabel('Edad (escalada)')
plt.ylabel('Poder Económico (escalado)')
plt.show()






