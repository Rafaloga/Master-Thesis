import os
from PIL import Image
import numpy as np
import json
import pycocotools.mask as maskUtils
def apply_mask_to_image(image, mask):
    """
    Applies the mask to the image to set the background to white for a grayscale image.
    """
    white_background = np.full(image.shape, 255, dtype=np.uint8)  # white background for grayscale image
    return np.where(mask == 0, white_background, image)



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
# Directorio donde se encuentran las imágenes originales
# Directorio donde se encuentran las imágenes originales
input_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/GX010042/000100'

# Directorio donde se guardarán las imágenes binarizadas
output_directory = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops_binary/GX010042/000100'

# Ruta al archivo JSON de anotaciones
annotations_file = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation/000100_semantic.json'

# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Listar todos los archivos en el directorio de entrada
image_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

# Cargar las anotaciones desde el archivo JSON
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Obtener el tamaño de la imagen completa
image_height, image_width = annotations["annotations"][0]["segmentation"]["size"]

# Crear una imagen en blanco del tamaño completo
full_binary_image = np.zeros((image_height, image_width), dtype=np.uint8)

# Iterar a través de las anotaciones y agregar cada imagen de recorte binarizada a la imagen completa
for i, annotation in enumerate(annotations["annotations"]):
    # Obtener la posición de la imagen de recorte en la imagen completa
    x, y, w, h = annotation["bbox"]

    # Construir el nombre del archivo de imagen binarizada en función del índice
    image_file_name = f'binarized_recorte_{i}.jpg'
    image_path = os.path.join(output_directory, image_file_name)
    binary_image = np.array(Image.open(image_path))

    # Agregar la imagen de recorte a la imagen completa en la posición correcta
    full_binary_image[y:y + h, x:x + w] = binary_image

# Guardar la imagen completa binarizada

# Crear la máscara
# Asumiendo que todas las anotaciones tienen el mismo tamaño que la imagen original
image_size = full_binary_image.shape[:2]
mask = create_single_channel_image(annotations, image_size)

# Aplicar la máscara a la imagen
masked_image = apply_mask_to_image(full_binary_image, mask)
# Aplicar la máscara a la imagen binaria para establecer el fondo en blanco

output_path = os.path.join(output_directory, 'binarized_full_image.png')
Image.fromarray(masked_image).save(output_path)