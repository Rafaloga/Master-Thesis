import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json

def guardar_diccionario(diccionario, nombre_archivo):
    with open(nombre_archivo, 'w') as f:
        diccionario_convertido = {clave: valor.tolist() for clave, valor in diccionario.items()}
        json.dump(diccionario_convertido, f)

def cargar_diccionario(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        diccionario = json.load(f)
        diccionario_convertido = {clave: np.array(valor) for clave, valor in diccionario.items()}
        return diccionario_convertido

def asignar_colores(df, mapa_de_colores):
    df['Color'] = df['Género'].map(mapa_de_colores)
    return df

def mostrar_colores(mapa_de_colores):
    plt.figure(figsize=(12, 3))
    plt.title('Mapa de Colores por Clase')
    for i, (clase, color) in enumerate(mapa_de_colores.items()):
        plt.fill_betweenx([0, 1], i, i+1, color=color)
        plt.text(i + 0.5, 1.05, clase, ha='center', va='bottom', fontsize=10, rotation=90)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.show()

# Función para generar colores aleatorios
def generar_colores_aleatorios(n):
    return np.random.rand(n, 3)

# Leer los archivos CSV
df1 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Resultados/GX010042_classes.csv')
df2 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Resultados/video1_classes.csv')

# Combinar los DataFrames para encontrar todas las clases únicas
df_combinado = pd.concat([df1, df2])
clases_unicas = df_combinado['Género'].unique()

# Número de clases únicas
n_clases = len(clases_unicas)

# Generar colores aleatorios para todas las clases
colores_aleatorios = generar_colores_aleatorios(n_clases)

# Crear el mapa de colores
mapa_de_colores = dict(zip(clases_unicas, colores_aleatorios))

mostrar_colores(mapa_de_colores)

# Guardar el diccionario
guardar_diccionario(mapa_de_colores, 'mapa_de_colores_aux.json')

# Cargar el diccionario
mapa_de_colores_cargado = cargar_diccionario('mapa_de_colores_aux.json')
print(mapa_de_colores_cargado)
