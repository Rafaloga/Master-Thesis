import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2
import os
from matplotlib.image import imread
import time
import re
import csv

'''def crear_mapa_densidad(coordenadas_csv, densidad_csv, output_path, vmin, vmax, guardar_como='imagen'):
    coordenadas_df = pd.read_csv(coordenadas_csv)
    densidad_df = pd.read_csv(densidad_csv)


    fig, ax = plt.subplots(figsize=(10, 8))

    norm = Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        coordenadas_df['long'],
        coordenadas_df['lat'],
        c=densidad_df['Densidad'],
        cmap='YlGn',
        s=50,
        alpha=0.7,
        norm=norm
    )

    cbar = plt.colorbar(sc)
    cbar.set_label('Densidad')

    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Mapa de Densidad')

    if guardar_como == 'imagen':
        extension = 'png'
    elif guardar_como == 'pdf':
        extension = 'pdf'
    else:
        raise ValueError("El valor de 'guardar_como' debe ser 'imagen' o 'pdf'.")

    output_file = f'{output_path}mapa_densidad.{extension}'
    plt.savefig(output_file, dpi=300)

    print(f"Mapa de densidad guardado como '{output_file}'.")'''

def crear_mapa_densidad(coordenadas_csv, densidad_csv, output_path, vmin, vmax, guardar_como='imagen'):
    # Leer los archivos CSV
    coordenadas_df = pd.read_csv(coordenadas_csv)
    densidad_df = pd.read_csv(densidad_csv)

    # Renombrar columnas para facilitar el merge
    densidad_df = densidad_df.rename(columns={'Image Name': 'frame'})
    densidad_df['frame'] = densidad_df['frame'].str.rstrip('.jpg').astype(int)  # Convertir nombres de imagen a número de frame

    # Unir los DataFrames basándose en los frames
    merged_df = pd.merge(densidad_df, coordenadas_df, on='frame')

    # Crear el mapa de densidad
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = Normalize(vmin=vmin, vmax=1.0)

    sc = ax.scatter(
        merged_df['long'],
        merged_df['lat'],
        c=merged_df['Densidad'],
        cmap='YlGn',
        s=50,
        alpha=0.7,
        norm=norm
    )

    # Configurar la barra de color
    cbar = plt.colorbar(sc)
    cbar.set_label('Densidad')

    # Configurar etiquetas y título
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title('Mapa de Densidad')

    # Guardar la imagen
    if guardar_como == 'imagen':
        extension = 'png'
    elif guardar_como == 'pdf':
        extension = 'pdf'
    else:
        raise ValueError("El valor de 'guardar_como' debe ser 'imagen' o 'pdf'.")

    output_file = f'{output_path}mapa_densidad.{extension}'
    plt.savefig(output_file, dpi=300)

    print(f"Mapa de densidad guardado como '{output_file}'.")


def crear_mapas_individuales(coordenadas_csv, densidad_csv, vmin_global, vmax_global, colormap, output_path):
    # Cargar los DataFrames desde las rutas de los archivos CSV
    coordenadas_df = pd.read_csv(coordenadas_csv)
    densidad_df = pd.read_csv(densidad_csv)


    # Normalizar los valores de densidad para que estén en el rango [0, 1]
    norm = Normalize(vmin=vmin_global, vmax=vmax_global)

    # Listas para acumular los puntos y densidades
    longitudes_acum = []
    latitudes_acum = []
    densidades_acum = []

    for index, row in coordenadas_df.iterrows():
        print(index)
        frame = row['frame']
        latitud = row['lat']
        longitud = row['long']

        # Obtener la densidad correspondiente para este frame
        frame_str = f'{frame:06.0f}.jpg'
        print(frame_str)
        #densidad = densidad_df.loc[densidad_df['Image Name'] == f'{frame_str}_semantic', 'Densidad'].values[0]
        densidad = densidad_df.loc[densidad_df['Image Name'] == f'{frame_str}', 'Densidad'].values[0]

        # Añadir las coordenadas y densidad a las listas acumuladas
        longitudes_acum.append(longitud)
        latitudes_acum.append(latitud)
        densidades_acum.append(densidad)

        # Crear una figura que contenga el mapa, la coordenada en verde y la barra de colores
        fig, ax_mapa = plt.subplots(figsize=(10, 6))

        # Agregar el mapa completo como fondo con una línea negra
        ax_mapa.scatter(
            coordenadas_df['long'],
            coordenadas_df['lat'],
            c='gray',  # Color negro para la línea
            s=1,  # Tamaño de los puntos
            alpha=0.7,  # Transparencia de la línea
        )

        # Agregar un marcador para la coordenada del frame en el mapa (verde)
        sc = ax_mapa.scatter(
            longitudes_acum,
            latitudes_acum,
            c=densidades_acum,  # Utilizar la lista acumulada
            cmap=colormap,  # Utilizar el mismo colormap que en el mapa general
            marker='o',
            s=100,  # Tamaño del marcador
            zorder=2,  # Asegura que el punto verde esté por encima de la línea negra
            norm=norm,  # Normalizar los valores de densidad para el colormap
        )

        # Personalizar etiquetas de ejes y título para el mapa
        ax_mapa.set_xlabel('Longitud')
        ax_mapa.set_ylabel('Latitud')

        # Crear una barra de colores para la densidad en la figura (a la derecha)
        cbar = plt.colorbar(sc, ax=ax_mapa, orientation='vertical', pad=0.1)
        cbar.set_label('Densidad')
        densidad_normalizada = norm(densidad)
        cbar.ax.axhline(densidad_normalizada, color='red', linewidth=2)

        # Guardar el mapa individual como una imagen en la ubicación especificada
        output_file = f'{output_path}/mapa_frame_{int(frame)}.png'
        plt.savefig(output_file, dpi=300)

        # Cerrar la figura para liberar memoria
        plt.close()

def map_frames_to_files(img_dir):
    return {
        int(f.split('.')[0]): f
        for f in sorted(os.listdir(img_dir))
        if f.split('.')[0].isdigit() and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    }

def build_image_lists_from_csv(csv_file, img_dir1, img_dir2):
    # Crear diccionarios para mapear el número de frame a nombre de archivo
    frame_to_file1 = map_frames_to_files(img_dir1)
    frame_to_file2 = map_frames_to_files(img_dir2)

    # Leer el archivo CSV y construir las listas en base a las relaciones especificadas
    img_files1 = []
    img_files2 = []

    with open(csv_file, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            try:
                frame2 = int(row['Frame Video 1'])
                frame1 = int(row['Frame Video 2'])
            except ValueError:
                # Manejar el caso donde el valor no se puede convertir a int
                continue

            # Solo añadir a la lista si el frame existe en los diccionarios
            if frame1 in frame_to_file1 and frame2 in frame_to_file2:
                img_files1.append(frame_to_file1[frame1])
                img_files2.append(frame_to_file2[frame2])

    return img_files1, img_files2

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(0))
    return None
def representar_imagenes_mapas(img_dir1, img_dir2, map_dir1, map_dir2, csv_file, output_path):
    # Obtener la lista de nombres de archivos en los directorios

    '''img_files1 = sorted(os.listdir(img_dir1))
    img_files2 = sorted(os.listdir(img_dir2))
    map_files1 = sorted(os.listdir(map_dir1))
    map_files2 = sorted(os.listdir(map_dir2))'''
    img_files1 = sorted(os.listdir(img_dir1))
    img_files2 = sorted(os.listdir(img_dir2))
    map_files1 = sorted(os.listdir(map_dir1), key=extract_number)
    map_files2 = sorted(os.listdir(map_dir2), key=extract_number)

    # Assuming that 'Frame Video 1' and 'Frame Video 2' are the column headers in the CSV
    img_files1, img_files2 = build_image_lists_from_csv(csv_file, img_dir1, img_dir2)
    #map_files1, map_files2 = build_image_lists_from_csv(csv_file, map_dir1, map_dir2)


    # Asegurarse de que los directorios de salida existan
    os.makedirs(output_path, exist_ok=True)

    for i in range(len(img_files1)):
        # Marca temporal en el punto final
        #print(img_files1[i])
        img1_path = os.path.join(img_dir1, img_files1[i])
        # Extraer el número del archivo img1_path, suponiendo que siempre tendrá el formato de 6 dígitos y la extensión .jpg
        frame_number_with_extension = os.path.basename(img1_path)  # Esto sería '000000.jpg'
        frame_number = os.path.splitext(frame_number_with_extension)[0]  # Esto sería '000000'

        # Convertir la cadena de número en un entero para eliminar los ceros iniciales, y luego de vuelta a cadena
        frame_number_int = int(frame_number)  # Esto convierte '000000' a 0
        frame_number_str = str(frame_number_int)  # Esto convierte 0 de vuelta a '0'

        # Formatear la cadena para que tenga el formato deseado, en este caso 'mapa_frame_x.png'
        map_filename = f"mapa_frame_{frame_number_str}.png"

        # Crear la ruta completa concatenando map_dir1 y map_filename
        map1_path = os.path.join(map_dir1, map_filename)

        img2_path = os.path.join(img_dir2, img_files2[i])
        # Extraer el número del archivo img1_path, suponiendo que siempre tendrá el formato de 6 dígitos y la extensión .jpg
        frame_number_with_extension2 = os.path.basename(img2_path)  # Esto sería '000000.jpg'
        frame_number2 = os.path.splitext(frame_number_with_extension2)[0]  # Esto sería '000000'

        # Convertir la cadena de número en un entero para eliminar los ceros iniciales, y luego de vuelta a cadena
        frame_number_int2 = int(frame_number2)  # Esto convierte '000000' a 0
        frame_number_str2 = str(frame_number_int2)  # Esto convierte 0 de vuelta a '0'

        # Formatear la cadena para que tenga el formato deseado, en este caso 'mapa_frame_x.png'
        map_filename2 = f"mapa_frame_{frame_number_str2}.png"

        # Crear la ruta completa concatenando map_dir1 y map_filename
        map2_path = os.path.join(map_dir2, map_filename2)


        img2_path = os.path.join(img_dir2, img_files2[i])
        # Crear una figura con subplots para cada imagen y mapa
        fig, axs = plt.subplots(2, 2, figsize=(24, 16))

        # Subgráfico 1: Mapa 1

        img_path1 = os.path.join(map_dir1, map_files1[i])
        print(map1_path)
        axs[0, 0].imshow(imread(map1_path))
        axs[0, 0].axis('off')

        # Subgráfico 2: Imagen 1
        print(img1_path)
        axs[1, 0].imshow(imread(img1_path))
        axs[1, 0].axis('off')

        # Subgráfico 3: Mapa 2
        print(map2_path)
        axs[0, 1].imshow(imread(map2_path))
        axs[0, 1].axis('off')

        # Subgráfico 4: Imagen 2

        print(img2_path)
        axs[1, 1].imshow(imread(img2_path))
        axs[1, 1].axis('off')

        plt.subplots_adjust(top=0.1, bottom=0.01, left=0.01, right=0.10, hspace=0.25, wspace=0.01)

        plt.tight_layout()

        # Generar un nombre de archivo único para guardar la imagen
        output_file = os.path.join(output_path, f'imagen_mapa_{i}.jpg')
        print(output_file)
        inicio = time.time()
        # Guardar la figura compuesta con un nombre único
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        final = time.time()
        tiempo_transcurrido = final - inicio
        #print(f"Tiempo transcurrido: {tiempo_transcurrido} segundos")
        # Cerrar la figura para liberar memoria
        plt.close()




# Cargar los datos de coordenadas desde el primer archivo CSV
coordenadas_csv = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/GX010042-frame_gps_interp.csv'
output = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps_separada/GX010042/'
output1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps_separada/video1/'
# Cargar los datos de densidad desde el segundo archivo CSV
densidad_csv = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/densidad_selection/GX010042_pixel_counts_resultados.csv'
output_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/GX010042/'

# Cargar los datos de coordenadas desde el primer archivo CSV
coordenadas_csv1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/video1-frame_gps_reordered.csv'

# Cargar los datos de densidad desde el segundo archivo CSV
densidad_csv1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/densidad_selection/video1_pixel_counts_resultados.csv'
output_dir1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/video1/'
# Ejemplo de uso:
# Carga los conjuntos de datos
densidad_df_1 = pd.read_csv(densidad_csv)
densidad_df_2 = pd.read_csv(densidad_csv1)

# Encuentra los valores mínimo y máximo globales de densidad
vmin_global = min(densidad_df_1['Densidad'].min(), densidad_df_2['Densidad'].min())
vmax_global = max(densidad_df_1['Densidad'].max(), densidad_df_2['Densidad'].max())

vmin_global1 = densidad_df_1['Densidad'].min()
vmax_global1 = densidad_df_1['Densidad'].max()
vmin_global2 = densidad_df_2['Densidad'].min()
vmax_global2 = densidad_df_2['Densidad'].max()

# Ahora llama a la función crear_mapa_densidad con los valores vmin y vmax globales
#crear_mapa_densidad(coordenadas_csv, densidad_csv, output, vmin_global1, vmax_global1, guardar_como='imagen')
#crear_mapa_densidad(coordenadas_csv1, densidad_csv1, output1, vmin_global2, vmax_global2, guardar_como='imagen')
# Llamar a la función para crear mapas individuales
crear_mapas_individuales(coordenadas_csv, densidad_csv, vmin_global, vmax_global, colormap='YlGn', output_path=output_dir)
crear_mapas_individuales(coordenadas_csv1, densidad_csv1,  vmin_global, vmax_global, colormap='YlGn', output_path=output_dir1)

# Llamar a la función con los directorios de imágenes y mapas
img_dir1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img/'
img_dir2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/'
map_dir1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/GX010042/'
map_dir2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/video1/'
output_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/visualizador_combined/'
matched_frames_csv = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/matched_frames.csv'

#representar_imagenes_mapas(img_dir1, img_dir2, map_dir1, map_dir2, matched_frames_csv, output_path)