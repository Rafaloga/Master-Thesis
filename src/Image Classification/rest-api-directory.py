import os
import requests
import json
import sys

# Directorio raíz que contiene todas las carpetas con imágenes
root_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1/'

# Obtener una lista de todas las carpetas en el directorio raíz, ordenadas alfabéticamente
folders = sorted([folder for folder in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, folder))])

#folders = folders[:]

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
    'api-key': '2b10fRljjEAFXm6GM3zm5Tmmu'
}
data = {
    'organs': 'auto'
}

# Iterar a través de las carpetas
for folder in folders:
    # Directorio completo de la carpeta actual
    current_directory = os.path.join(root_directory, folder)

    # Obtener la lista de archivos de imágenes en la carpeta y ordenarla numéricamente
    image_files = sorted([f for f in os.listdir(current_directory) if f.endswith(".jpg")], key=lambda x: int(''.join(filter(str.isdigit, x))))

    # Diccionario para almacenar los resultados con los nombres de las imágenes como clave
    results_dict = {}

    # Variable para verificar si se ha alcanzado el límite de solicitudes
    limit_exceeded = False

    #image_files = image_files[32:]
    # Iterar a través de los archivos en orden y mantener el nombre de la imagen
    for filename in image_files:
        image_path = os.path.join(current_directory, filename)
        files = {
            'images': (filename, open(image_path, 'rb'), 'image/jpeg')
        }

        response = requests.post(url, headers=headers, params=params, files=files, data=data)

        # Definir response_json aquí para que esté disponible en todo el bloque
        response_json = response.json()

        if response.status_code == 200:
            # Obtener las tres clases más probables con sus scores
            top_classes = response_json.get('results', [])[:3]

            # Guardar los resultados en el diccionario usando el nombre de la imagen como clave
            results_dict[filename] = {
                'top_classes': top_classes
            }

            # Guardar los resultados en un archivo JSON específico para esta carpeta
            json_filename = f"/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/video1/{folder}_results.json"

            with open(os.path.join(current_directory, json_filename), 'w') as json_file:
                json.dump(results_dict, json_file, indent=4)

            print(f"Resultados para la carpeta '{folder} / {filename}' guardados en '{json_filename}'")
        elif response.status_code == 429:
            error_message = response_json.get('message', '')
            print(f"Error en la solicitud para {folder} / {filename}: Código de estado {response.status_code}, Mensaje de error: {error_message}")
            sys.exit()  # Salir del script por completo
            # Verificar si el mensaje de error indica límite de solicitudes excedido

        else:
            response_message = response_json.get('message', '')
            print(
                f"Imagen {folder} / {filename}. Código de estado {response.status_code}, Mensaje: {response_message}")
