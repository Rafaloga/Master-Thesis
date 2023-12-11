import cv2
import pandas as pd
import os
# Leer el archivo CSV
df = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/matched_frames/matched_frames_euclidean_corrected.csv')

# Directorios donde están los frames de cada video
directorio_video1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
directorio_video2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/'

# Crear los objetos de video de salida
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# Asegúrate de reemplazar las dimensiones (640, 480) por las de tus frames
out1 = cv2.VideoWriter('video_euclidean_procesado1_corrected.mp4', fourcc, 24.0, (1920, 1080))
out2 = cv2.VideoWriter('video_euclidean_procesado2_corrected.mp4', fourcc, 24.0, (1920, 1080))

# Procesar cada frame
for index, row in df.iterrows():
    print(index)
    # Formatear el nombre del frame con ceros a la izquierda
    frame1_name = str(int(row['Frame Video 1'])).zfill(6) + '.jpg'
    frame2_name = str(int(row['Frame Video 2'])).zfill(6) + '.jpg'

    frame1_path = os.path.join(directorio_video1, frame1_name)
    frame2_path = os.path.join(directorio_video2, frame2_name)

    # Leer y añadir frame del video 1
    frame1 = cv2.imread(frame1_path)
    if frame1 is not None:
        out1.write(frame1)

    # Leer y añadir frame del video 2
    frame2 = cv2.imread(frame2_path)
    if frame2 is not None:
        out2.write(frame2)

# Liberar recursos
out1.release()
out2.release()
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

df = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/geo_matched_frames.csv')

# Directorios donde están los frames de cada video
dir_video2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
dir_video1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/'

# Función para formatear el nombre del archivo según el número de frame
def format_filename(frame_number):
    return f"{int(frame_number):06d}.jpg"

# Preparar la figura y los ejes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
plt.ion()  # Activar modo interactivo

# Función para actualizar y mostrar dos imágenes lado a lado
def show_frames(row):
    frame_video1 = os.path.join(dir_video1, format_filename(row['Frame Video 1']))  # Asumiendo formato jpg
    frame_video2 = os.path.join(dir_video2, format_filename(row['Frame Video 2']))
    print(frame_video1)
    print(frame_video2)
    img1 = Image.open(frame_video1)
    img2 = Image.open(frame_video2)

    # Actualizar imágenes
    axes[0].imshow(img1)
    axes[0].set_title('Video 1 Frame')
    axes[0].axis('off')

    axes[1].imshow(img2)
    axes[1].set_title('Video 2 Frame')
    axes[1].axis('off')

    plt.draw()
    plt.pause(1/24)  # Pausa de 0.5 segundos entre imágenes

# Iterar sobre cada fila del DataFrame y mostrar las imágenes
for index, row in df.iterrows():
    show_frames(row)

plt.ioff()  # Desactivar modo interactivo
plt.show()'''