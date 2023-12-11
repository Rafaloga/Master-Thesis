import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Directorio donde se encuentran las imágenes originales
input_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/GX010042/000100'

# Directorio donde se guardarán las imágenes binarizadas
output_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops_binary/GX010042/000100'

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Listar todos los archivos en el directorio de entrada
image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

# Procesar cada archivo de imagen
for image_file in image_files:
    # Leer la imagen
    image_path = os.path.join(input_directory, image_file)
    image = np.array(Image.open(image_path))

    # Convertir a escala de grises
    gray_image = rgb2gray(image)

    # Calcular el umbral de Otsu
    otsu_threshold = threshold_otsu(gray_image)

    # Binarizar la imagen según el umbral de Otsu
    binary_image = gray_image > otsu_threshold

    # Convertir de nuevo a formato de imagen para guardar
    binary_image_formatted = (binary_image * 255).astype(np.uint8)

    # Invertir la imagen si es necesario
    # Esto dependerá de cómo quieras tus resultados
    # inverted_binary_image = ~binary_image_formatted

    # Guardar la imagen binarizada
    output_path = os.path.join(output_directory, f'binarized_{image_file}')
    Image.fromarray(binary_image_formatted).save(output_path)

    # Imprimir la cantidad de píxeles blancos (opcional)
    count_zeros = np.sum(binary_image == 0)
    count_ones = np.sum(binary_image == 1)
    print(f'Image: {image_file}, Count of black pixels: {count_zeros}')
    print(count_ones)
