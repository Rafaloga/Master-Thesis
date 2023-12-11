import os
import json
import pandas as pd


def obtener_recortes_filtrados(csv_ruta):
    # Leer el archivo CSV y obtener los recortes a filtrar
    df = pd.read_csv(csv_ruta)

    # Crear un diccionario vacío
    recortes_filtrados = {}

    # Iterar sobre cada fila del DataFrame
    for index, row in df.iterrows():
        # Convertir la cadena de 'mixed_list' en una lista de enteros
        #mixed_list = [int(i.strip()) for i in row['mixed_list'].strip("[]").split(",")]
        mixed_list = [int(i.strip()) for i in row['mixed_list'].strip("[]").split(",") if i.strip()]
        # Obtener el nombre de la imagen sin la extensión '.jpg'
        img_key = os.path.splitext(row['img'])[0]

        # Asignar al diccionario usando como clave el nombre de la imagen sin la extensión
        recortes_filtrados[img_key] = mixed_list

    return recortes_filtrados


def procesar_directorio(directorio, carpeta_destino, csv_ruta):
    datos = []
    recortes_filtrados = obtener_recortes_filtrados(csv_ruta)
    print(recortes_filtrados)
    archivos_json = [archivo for archivo in os.listdir(directorio) if archivo.endswith("_results.json")]
    archivos_json_ordenados = sorted(archivos_json, key=lambda x: int(x.split('_')[0]))

    for archivo in archivos_json_ordenados:
        numero_frame = archivo.split('_')[0]
        print(numero_frame)
        if numero_frame not in recortes_filtrados:
            continue  # Si el número de frame no está en los recortes filtrados, no procesar este archivo

        ruta_completa = os.path.join(directorio, archivo)

        with open(ruta_completa, "r") as f:
            contenido = json.load(f)

        # Filtra los recortes basado en la información del archivo CSV
        recortes_a_procesar = recortes_filtrados[numero_frame]
        print(recortes_a_procesar)

        for recorte in recortes_a_procesar:
            recorte_nombre = f"recorte_{recorte}.jpg"

            info = contenido.get(recorte_nombre)

            if info and "top_classes" in info and info["top_classes"]:
                primera_clase = info["top_classes"][0]
                nombre_cientifico = primera_clase["species"]["genus"]["scientificNameWithoutAuthor"]
                score = primera_clase["score"]
                datos.append([numero_frame, recorte_nombre, nombre_cientifico, score])
            elif info and "error" in info:
                datos.append([numero_frame, recorte_nombre, info["error"], None])
            else:
                datos.append([numero_frame, recorte_nombre, "No especificado", None])

    df = pd.DataFrame(datos, columns=["Frame", "Recorte", "Nombre Científico", "Score"])
    nombre_archivo = os.path.basename(os.path.normpath(directorio)) + ".csv"
    ruta_archivo_destino = os.path.join(carpeta_destino, nombre_archivo)
    df.to_csv(ruta_archivo_destino, index=False)


# Ejemplo de uso

dir1 = "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/GX010042/"
dir2 = "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/video1/"
csv1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_GX010042.csv'
csv2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv'
carpeta_destino = "/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/results/filtrado/"


directorios_y_csv = {
    dir1: csv1,
    dir2: csv2,

}



for directorio, csv_ruta in directorios_y_csv.items():
    procesar_directorio(directorio, carpeta_destino, csv_ruta)