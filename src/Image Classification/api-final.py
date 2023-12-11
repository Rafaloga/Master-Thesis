import os
import requests
import json
import sys
import csv
import re
import cv2
import numpy as np


# Definir la URL y otros parámetros para la API
url = 'https://my-api.plantnet.org/v2/identify/all'
headers = {
    'accept': 'application/json'
}
params = {
    'include-related-images': 'false',
    'no-reject': 'false',
    'lang': 'en',
    'type': 'kt',
    'api-key': '2b10UiFup25U2llQHbJW4m7OKu'
}
data = {
    'organs': 'auto'
}
def get_folders(directory):
    return sorted([folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))])

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def get_image_files(directory, crops_list):
    files = sorted([f for f in os.listdir(directory) if f.endswith(".jpg")], key=extract_number)
    return [f for i, f in enumerate(files) if i in crops_list]

def post_image_to_api(image_path, filename):
    with open(image_path, 'rb') as image_file:
        files = {'images': (filename, image_file, 'image/jpeg')}
        response = requests.post(url, headers=headers, params=params, files=files, data=data)
    return response

def save_results_to_json(results, json_filename):
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

def cargar_crops_csv(archivo_csv):
    crops_dict = {}
    with open(archivo_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            nombre_carpeta, lista_crops = row[0].replace('.jpg', ''), eval(row[1])
            #print(nombre_carpeta)
            crops_dict[nombre_carpeta] = lista_crops
            #break
    return crops_dict

def process_folder(folder, crops_dict, root_directory, json_filename):
    current_directory = os.path.join(root_directory, folder)
    image_files = get_image_files(current_directory, crops_dict[folder])
    results_dict = {}

    print(f"Procesando carpeta: {folder}")  # Añadido mensaje de inicio de procesamiento de la carpeta

    for filename in image_files:
        #print(f"Procesando imagen: {filename}")  # Añadido mensaje de inicio de procesamiento de imagen
        image_path = os.path.join(current_directory, filename)
        response = post_image_to_api(image_path, filename)

        if response.status_code == 200:
            top_classes = response.json().get('results', [])
            results_dict[filename] = {'top_classes': top_classes}
            print(f"Imagen {filename} procesada con éxito.")  # Añadido mensaje de éxito
        else:
            error_message = response.json().get('message', '')
            print(f"Error para la imagen {filename}: {response.status_code}, Mensaje: {error_message}")  # Añadido mensaje de error
            results_dict[filename] = {'error': error_message}

    save_results_to_json(results_dict, json_filename)
    print(f"Resultados para la carpeta '{folder}' guardados en '{json_filename}'")  # Mensaje final para la carpeta

# Creamos una nueva función que procesa un conjunto de datos completo
def process_alternating_folders(root_directory1, archivo_csv1, root_directory2, archivo_csv2):
    crops_a_procesar1 = cargar_crops_csv(archivo_csv1)
    crops_a_procesar2 = cargar_crops_csv(archivo_csv2)
    folders1 = get_folders(root_directory1)
    folders2 = get_folders(root_directory2)

    indice_inicio = 6913
    # Convertir las claves del diccionario a una lista para mantener el orden
    claves_ordenadas1 = list(crops_a_procesar1.keys())
    # Filtrar el diccionario basado en el índice
    crops_a_procesar1 = {clave: crops_a_procesar1[clave] for clave in claves_ordenadas1[indice_inicio:]}
    claves_ordenadas2 = list(crops_a_procesar2.keys())
    # Filtrar el diccionario basado en el índice
    crops_a_procesar2 = {clave: crops_a_procesar2[clave] for clave in claves_ordenadas2[indice_inicio:]}
    #print(crops_a_procesar1)
    for folder1, folder2 in zip(folders1, folders2):
        if folder1 in crops_a_procesar1:
            # Procesar carpeta del primer directorio
            json_filename1 = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/final/{os.path.basename(os.path.normpath(root_directory1))}/{folder1}_results.json"
            process_folder(folder1, crops_a_procesar1, root_directory1, json_filename1)

        if folder2 in crops_a_procesar2:
            #print(folder2)
            # Procesar carpeta del segundo directorio
            json_filename2 = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/final/{os.path.basename(os.path.normpath(root_directory2))}/{folder2}_results.json"
            process_folder(folder2, crops_a_procesar2, root_directory2, json_filename2)


# Llamada a la función para procesar alternadamente
process_alternating_folders(
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/GX010042/',
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_GX010042.csv',
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1/',
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_video1.csv'
)






