import pandas as pd
import os

# Lista de nombres de archivos CSV que contienen la cantidad de píxeles de vegetación
archivo1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/pixels_selection/GX010042_pixel_counts.csv'
archivo2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/pixels_selection/video1_pixel_counts_aux.csv'
archivos_csv = [archivo1, archivo2]  # Agrega los nombres de tus archivos CSV

resultados_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/densidad_selection_aux'  # Directorio donde se guardarán los resultados

# Inicializa una variable para almacenar el valor máximo de píxeles
max_pixels_value = 0
max_pixels_index = None

# Itera a través de los archivos CSV para encontrar el máximo de todos los videos
for archivo in archivos_csv:
    df = pd.read_csv(archivo)
    max_frame_index = df['Total Pixels'].idxmax()
    max_frame_value = df.at[max_frame_index, 'Total Pixels']

    if max_frame_value > max_pixels_value:
        max_pixels_value = max_frame_value
        max_pixels_index = max_frame_index
        print('Max value is', max_pixels_value, 'in frame', max_frame_index, 'of video', archivo)

# Itera nuevamente a través de los archivos CSV para calcular la densidad
for archivo in archivos_csv:
    # Leer el archivo CSV
    df = pd.read_csv(archivo)

    # Calcular la densidad de píxeles para cada fila en el DataFrame original
    df['Densidad'] = df['Total Pixels'] / max_pixels_value
    #df['Densidad'] = df['Total Pixels'] / 2073600
    # Obtener el nombre del video
    video_name = os.path.splitext(os.path.basename(archivo))[0]

    # Crear un directorio para guardar los resultados si no existe
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)

    # Guardar los resultados en un nuevo archivo CSV con el nombre del video
    resultado_archivo = os.path.join(resultados_dir, f'{video_name}_resultados.csv')
    df.to_csv(resultado_archivo, index=False)
