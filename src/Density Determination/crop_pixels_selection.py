import os
import csv
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

# Directorios
main_input_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1'
csv_file = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/pixels_selection/video1_pixel_counts_aux.csv'
csv_input_file = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_video1.csv'  # Añade la ruta a tu archivo CSV aquí

# Función para leer los crops del archivo CSV
def read_csv_crop_info(csv_path):
    crops_info = []
    with open(csv_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            img_name = row['img']
            mixed_list = eval(row['mixed_list'])
            crops_info.append((img_name, mixed_list))
    return crops_info

# Leer información de crops del archivo CSV manteniendo duplicados
crops_to_process = read_csv_crop_info(csv_input_file)
print(len(crops_to_process))
# Lista para guardar la información de los píxeles negros
pixel_info = []

# Procesar cada archivo de imagen especificado en el CSV
for img_info in crops_to_process:
    img_name, crop_list = img_info

    input_directory = os.path.join(main_input_directory, img_name.split('.')[0])

    total_black_pixels = 0

    for crop_number in crop_list:
        image_file = f"recorte_{crop_number}.jpg"

        if image_file in os.listdir(input_directory):
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
    pixel_info.append({'Image Name': img_name, 'Total Pixels': total_black_pixels})
    print('Image Name:', img_name, 'Total Pixels:', total_black_pixels)

# Guardar la información en un archivo CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['Image Name', 'Total Pixels'])
    writer.writeheader()
    for row in pixel_info:
        writer.writerow(row)

print(f"Finished processing. The pixel counts have been saved to {csv_file}")
