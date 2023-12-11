import os
import csv
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Directorio principal donde se encuentran los subdirectorios de las imágenes originales
main_input_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1'


# Archivo CSV para guardar el conteo de píxeles negros
csv_file = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/pixels/video1_pixel_counts.csv'

# Lista para guardar la información de los píxeles negros
pixel_info = []
directory_files = sorted(os.listdir(main_input_directory))
# Iterar sobre cada subdirectorio en el directorio principal
for subdir in directory_files:
    input_directory = os.path.join(main_input_directory, subdir)




    total_black_pixels = 0

    # Listar todos los archivos en el directorio de entrada
    image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # Procesar cada archivo de imagen
    for image_file in image_files:
        # Leer la imagen
        image_path = os.path.join(input_directory, image_file)
        image = np.array(Image.open(image_path))

        # Convertir a escala de grises y calcular el umbral de Otsu
        gray_image = rgb2gray(image)
        otsu_threshold = threshold_otsu(gray_image)

        # Binarizar la imagen y guardar
        binary_image = gray_image > otsu_threshold
        binary_image_formatted = (binary_image * 255).astype(np.uint8)


        # Contar la cantidad de píxeles negros y acumular
        count_zeros = np.sum(binary_image == 0)
        total_black_pixels += count_zeros

    # Añadir la información del subdirectorio al CSV
    pixel_info.append({'Image Name': subdir, 'Total Pixels': total_black_pixels})
    print('Image Name:', subdir, 'Total Pixels:', total_black_pixels)
# Guardar la información en un archivo CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Image Name', 'Total Pixels'])
    writer.writeheader()
    for row in pixel_info:
        writer.writerow(row)

print(f"Finished processing. The pixel counts have been saved to {csv_file}")