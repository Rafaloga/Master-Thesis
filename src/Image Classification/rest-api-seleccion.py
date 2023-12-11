import os
import requests
import json
import sys
import csv
import re

# Directorio raíz que contiene todas las carpetas con imágenes
root_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1/'

# Obtener una lista de todas las carpetas en el directorio raíz, ordenadas alfabéticamente
folders = sorted([folder for folder in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder))])

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
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None
# Cargar la lista de crops del archivo CSV
def cargar_crops_csv(archivo_csv):
    crops_dict = {}
    with open(archivo_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Saltar la cabecera
        for row in reader:
            nombre_carpeta, lista_crops = row[0].replace('.jpg', ''), eval(row[1])
            crops_dict[nombre_carpeta] = lista_crops
    return crops_dict

archivo_csv = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv'
crops_a_procesar = cargar_crops_csv(archivo_csv)

# Iterar a través de las carpetas
for folder in folders:
    if folder in crops_a_procesar:
        # Directorio completo de la carpeta actual
        current_directory = os.path.join(root_directory, folder)

        # Ordenar los archivos de imágenes basándote en los números dentro de los nombres de archivo
        image_files = sorted([f for f in os.listdir(current_directory) if f.endswith(".jpg")], key=extract_number)

        # Filtrar la lista de imágenes para procesar solo las especificadas en el archivo CSV
        image_files = [f for i, f in enumerate(image_files) if i in crops_a_procesar[folder]]
        print(image_files)
        # Diccionario para almacenar los resultados con los nombres de las imágenes como clave
        results_dict = {}

        # Diccionario para almacenar los resultados con los nombres de las imágenes como clave
        results_dict = {}

        # Iterar a través de los archivos en orden y mantener el nombre de la imagen
        for filename in image_files:
            image_path = os.path.join(current_directory, filename)
            files = {
                'images': (filename, open(image_path, 'rb'), 'image/jpeg')
            }

            response = requests.post(url, headers=headers, params=params, files=files, data=data)

            # Cerrar el archivo después de enviarlo
            files['images'][1].close()

            # Definir response_json aquí para que esté disponible en todo el bloque
            response_json = response.json()

            # Guardar los resultados en un archivo JSON específico para esta carpeta
            json_filename = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/video1_seleccion/{folder}_results.json"

            if response.status_code == 200:
                # Obtener las tres clases más probables con sus scores
                top_classes = response_json.get('results', [])[:3]
                results_dict[filename] = {'top_classes': top_classes}
            elif response.status_code == 429:
                error_message = response_json.get('message', '')
                print(
                    f"Error en la solicitud para {folder} / {filename}: Código de estado {response.status_code}, Mensaje de error: {error_message}")
                sys.exit()  # Salir del script por completo
            elif response.status_code == 404:
                response_message = response_json.get('message', '')
                print(
                    f"Imagen {folder} / {filename}. Código de estado {response.status_code}, Mensaje: {response_message}")
                results_dict[filename] = {'error': response_message}
            else:
                response_message = response_json.get('message', '')
                print(
                    f"Imagen {folder} / {filename}. Código de estado {response.status_code}, Mensaje: {response_message}")
                results_dict[filename] = {'error': response_message}

            with open(json_filename, 'w') as json_file:
                json.dump(results_dict, json_file, indent=4)

            print(f"Resultados para la carpeta '{folder} / {filename}' guardados en '{json_filename}'")
