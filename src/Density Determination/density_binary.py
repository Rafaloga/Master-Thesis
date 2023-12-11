import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pycocotools.mask as maskUtils
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
import csv
def apply_mask_to_image(image, mask):
    """
    Aplica la máscara a la imagen para poner el fondo en negro.
    """
    # Asumiendo que la máscara es una matriz binaria con los mismos
    # dimensiones que la imagen.
    for c in range(3):
        image[:, :, c] = np.where(mask == 0, 0, image[:, :, c])
    return image


def calculate_masked_pixel_sum(image, mask):
    """
    Calcula la suma de los valores de los píxeles de la imagen que están marcados por la máscara binaria.
    """
    # Asegúrate de que la máscara es una matriz booleana
    mask_bool = mask.astype(bool)

    # Sumar los valores de los píxeles en los canales RGB, teniendo en cuenta solo aquellos píxeles donde la máscara es True
    pixel_sum = np.sum(image[mask_bool])

    return pixel_sum


def create_single_channel_image(data, image_size):
    """
    Crea una imagen de canal único usando la segmentación.
    """
    mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    for ann in data['annotations']:
        # Aquí se decodifica la máscara de segmentación.
        bitmask = maskUtils.decode(ann['segmentation'])
        # Aplicar la máscara para obtener el objeto de interés.
        mask[bitmask > 0] = 255  # Usamos 255 para visualizar la máscara
    return mask


def calculate_black_pixel_sum(final_image):
    """
    Calcula la suma de los valores de los píxeles negros en la imagen final.
    """
    # Considerar solo los píxeles negros (valor 0)
    black_pixels = final_image == 0

    # Sumar los valores de los píxeles negros
    black_pixel_sum = black_pixels.sum()

    return black_pixel_sum


def apply_mask_to_binary_image(binary_image, mask):
    """
    Aplica la máscara a la imagen binaria para poner las zonas fuera de la máscara en blanco.
    """
    # Asumiendo que la máscara es una matriz binaria con los mismos
    # dimensiones que la imagen y que los objetos están marcados con 1s y el fondo con 0s.

    # Invertir la máscara: lo que era 0 será True y lo que era 1 será False
    inverted_mask = mask == 0

    # Aplicar la máscara invertida a la imagen binaria:
    # Donde la máscara invertida es True (originalmente 0 en la máscara), se establece el valor a 1 (blanco).
    # Donde la máscara invertida es False (originalmente 1 en la máscara), se mantiene el valor original de la imagen binaria.
    masked_binary_image = np.where(inverted_mask, 1, binary_image)

    return masked_binary_image

# Define el directorio donde están las imágenes y los JSON

json_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation/'
image_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
#json_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation/'
#image_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img'

# Obtén la lista de archivos json y las imágenes
json_files = sorted([file for file in os.listdir(json_dir) if file.endswith('_semantic.json')])
image_files = sorted([file for file in os.listdir(image_dir) if file.endswith('.jpg')])
# Nombre del archivo CSV de salida
csv_file = 'GX010042_pixels.csv'

# Encabezados del archivo CSV
headers = ["Image Name", "Total Pixels"]

# Abrir el archivo CSV para escritura
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)

    # Escribir los encabezados
    writer.writeheader()
    for json_file, image_file in zip(json_files, image_files):
        print(json_file)
        json_path = os.path.join(json_dir, json_file)
        image_path = os.path.join(image_dir, image_file)

        # Leer la imagen
        image = np.array(Image.open(image_path))

        # Leer el JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Asumiendo que todas las anotaciones tienen el mismo tamaño que la imagen original
        image_size = image.shape[:2]  # tamaño de la imagen: (alto, ancho)

        # Crear la máscara
        mask = create_single_channel_image(data, image_size)

        # Aplicar la máscara a la imagen
        masked_image = apply_mask_to_image(image, mask)

        # Convertir la imagen resultante a escala de grises para la umbralización de Otsu
        #gray_image = rgb2gray(masked_image)
        gray_image = rgb2gray(image)

        # Calcular el umbral de Otsu
        otsu_threshold = threshold_otsu(gray_image)

        # Binarizar la imagen según el umbral de Otsu
        binary_image = gray_image > otsu_threshold

        # Aplicar la máscara a la imagen binaria para establecer el fondo en blanco
        binary_image_with_mask = apply_mask_to_binary_image(binary_image, mask)



        count_zeros = np.sum(binary_image_with_mask == 0)
        print(count_zeros)
        # Aquí, escribe directamente en el archivo CSV
        writer.writerow({
            "Image Name": os.path.splitext(image_file)[0],  # quita la extensión del archivo
            "Total Pixels": count_zeros
            # Añade aquí más datos si los tienes
        })

        # Para guardar la imagen final después de aplicar la máscara de Otsu
        # Mostrar la imagen binarizada con la máscara aplicada
        '''fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 fila, 2 columnas
    
        # Configura la primera subgráfica
        axs[0].imshow(binary_image_with_mask, cmap='gray')
        axs[0].axis('off')  # No mostrar los ejes
    
        # Configura la segunda subgráfica
        axs[1].imshow(masked_image, cmap='gray')
        axs[1].axis('off')  # No mostrar los ejes
    
        # Ajusta el espacio entre las imágenes si es necesario
        plt.tight_layout(pad=0)
    
        # Guarda la figura completa con ambas imágenes
        plt.savefig('final_images_combined.png', bbox_inches='tight', pad_inches=0)
    
        # Muestra la figura con las dos subgráficas
        plt.show()# Cierra la figura para que no se muestre en pantalla'''
