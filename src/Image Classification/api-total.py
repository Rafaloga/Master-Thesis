import os
import requests
import json
import sys
import csv
import re
import cv2
import numpy as np
import pandas as pd


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
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(0))
    return None

def cargar_indices_csv(archivo_csv, video):
    # Leer el archivo CSV usando pandas

    df = pd.read_csv(archivo_csv)
    if video == 'video1':
    # Convertir los valores de la columna 'Frame Video 1' a cadenas de caracteres de seis dígitos
        indices = df['Frame Video 1'].apply(lambda x: str(int(x)).zfill(6)).tolist()
    elif video == 'video2':
        indices = df['Frame Video 2'].apply(lambda x: str(int(x)).zfill(6)).tolist()

    return indices

def get_folders(directory, archivo_csv, video):

    # Cargar los índices de las carpetas a procesar desde el CSV
    indices_a_procesar = cargar_indices_csv(archivo_csv, video)
    # Obtener todas las carpetas en el directorio
    #all_folders = sorted([folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))])
    # Filtrar y devolver solo las carpetas que coinciden con los índices del CSV
    # Obtener todas las carpetas en el directorio
    all_folders = []
    i = 0
    for folder in os.listdir(directory):
        full_path = os.path.join(directory, folder)
        if os.path.isdir(full_path):
            all_folders.append(folder)
            i+=1
            print("Valor de la iteración actual:", i)

    all_folders = sorted(all_folders)
    return [folder for folder in all_folders if folder in indices_a_procesar]



def get_image_files(current_directory):
    # Suponiendo que estás buscando archivos con estas extensiones de imagen comunes
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"]
    # Lista y filtra los archivos, y luego los ordena
    image_files = sorted([f for f in os.listdir(current_directory)
                          if os.path.splitext(f)[1].lower() in valid_image_extensions], key=extract_number)
    return image_files

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
            crops_dict[nombre_carpeta] = lista_crops
    return crops_dict

def process_folder(folder, root_directory, json_filename):
    current_directory = os.path.join(root_directory, folder)
    image_files = get_image_files(current_directory)  # Ahora obtiene todos los archivos de imagen
    results_dict = {}

    print(f"Procesando carpeta: {folder}")  # Mensaje de inicio de procesamiento de la carpeta

    for filename in image_files:
        image_path = os.path.join(current_directory, filename)
        response = post_image_to_api(image_path, filename)

        if response.status_code == 200:
            top_classes = response.json().get('results', [])[:3]
            results_dict[filename] = {'top_classes': top_classes}
            print(f"Imagen {filename} procesada con éxito.")  # Mensaje de éxito
        else:
            error_message = response.json().get('message', '')
            print(f"Error para la imagen {filename}: {response.status_code}, Mensaje: {error_message}")  # Mensaje de error
            results_dict[filename] = {'error': error_message}

    save_results_to_json(results_dict, json_filename)
    print(f"Resultados para la carpeta '{folder}' guardados en '{json_filename}'")  # Mensaje final para la carpeta

# Creamos una nueva función que procesa un conjunto de datos completo
def process_alternating_folders(root_directory1, root_directory2, archivo_csv):

    # Ya no necesitas cargar crops desde CSV, asumiendo que get_folders ya lo hace
    folders1 = get_folders(root_directory1, archivo_csv, 'video2')
    folders2 = get_folders(root_directory2, archivo_csv, 'video1')

    for folder1, folder2 in zip(folders1, folders2):
        # Procesar carpeta del primer directorio
        json_filename1 = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/{os.path.basename(os.path.normpath(root_directory1))}/{folder1}_results.json"
        process_folder(folder1, root_directory1, json_filename1)

        # Procesar carpeta del segundo directorio
        json_filename2 = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/{os.path.basename(os.path.normpath(root_directory2))}/{folder2}_results.json"
        process_folder(folder2, root_directory2, json_filename2)


# Llamada a la función para procesar alternadamente
process_alternating_folders(
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/GX010042/',
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1/',
    '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/matched_frames.csv',
    )






