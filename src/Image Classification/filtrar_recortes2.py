import os
import json
import pandas as pd

def procesar_directorios(directorios, carpeta_destino):
    for directorio in directorios:
        procesar_directorio(directorio, carpeta_destino)

def procesar_directorio(directorio, carpeta_destino):
    # Lista para almacenar los datos
    datos = []

    # Obtiene la lista de archivos en el directorio y los ordena
    archivos_json = [archivo for archivo in os.listdir(directorio) if archivo.endswith("_results.json")]
    archivos_json_ordenados = sorted(archivos_json, key=lambda x: int(x.split('_')[0]))

    # Recorre todos los archivos en el directorio
    for archivo in archivos_json_ordenados:
        if archivo.endswith("_results.json"):
            # Extrae el número del frame del nombre del archivo
            numero_frame = archivo.split('_')[0]

            ruta_completa = os.path.join(directorio, archivo)

            # Abre y lee el archivo .json
            with open(ruta_completa, "r") as f:
                contenido = json.load(f)

            # Extrae la información deseada
            for recorte, info in contenido.items():
                if "top_classes" in info and info["top_classes"]:  # Verifica que haya al menos una clase
                    # Toma solo la primera clase (la de score más alto)
                    primera_clase = info["top_classes"][0]
                    nombre_cientifico = primera_clase["species"]["genus"]["scientificNameWithoutAuthor"]
                    score = primera_clase["score"]  # Añade el puntaje
                    datos.append([numero_frame, recorte, nombre_cientifico, score])
                elif "error" in info:
                    datos.append([numero_frame, recorte, info["error"], None])  # Agrega None para el puntaje si hay un error
                else:
                    datos.append([numero_frame, recorte, "No especificado", None])  # Agrega None para el puntaje si no está especificado

    # Crea un DataFrame con los datos
    df = pd.DataFrame(datos, columns=["Frame", "Recorte", "Nombre Científico", "Score"])

    # Guarda los resultados en un archivo .csv en la carpeta de destino
    nombre_archivo = os.path.basename(os.path.normpath(directorio)) + ".csv"
    ruta_archivo_destino = os.path.join(carpeta_destino, nombre_archivo)
    df.to_csv(ruta_archivo_destino, index=False)

# Ejemplo de uso:
directorios_a_procesar = [
    "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/GX010042_seleccion/",
    "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/video1_seleccion/",
]
carpeta_destino = "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/filtrado/"

procesar_directorios(directorios_a_procesar, carpeta_destino)
