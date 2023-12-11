import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import random
import csv
import matplotlib.patches as mpatches


def transformar_puntos_bbox(img_annotations_path, homography_matrix=None, transformar=True, img_matchings=None):

    with open(img_annotations_path, 'r') as img_annotations_file:
        data = json.load(img_annotations_file)

    annotations = data["annotations"]
    transformed_points = []

    if img_matchings is not None:
        # Filtra las anotaciones basadas en las claves en img_matchings
        annotations = [annotations[i] for i in img_matchings]

    for annotation in annotations:
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox

        pts = np.array(
            [[x_min, y_min], [x_min + width, y_min], [x_min + width, y_min + height], [x_min, y_min + height]],
            dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)

        if transformar:
            transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)
            transformed_points.append(transformed_pts)
        else:
            transformed_points.append(pts)
    return transformed_points



def dibujar_puntos1(img, puntos_y_color, idx, estilo='points', thickness=2,
                    iou_threshold=0.5):
    puntos, color = puntos_y_color
    # Definir un valor predeterminado para color_bgr, por ejemplo, un color por defecto o None
    #print(color)
    # Comprobar si el color es 'Color no encontrado'
    if color == 'Color no encontrado':
        return None, None
    else:
        # Si el color es encontrado, realizar la conversión de RGBA a BGR y escalar a [0, 255]
        color_bgr = tuple(int(c * 255) for c in reversed(color[:3]))
        # Aquí puedes continuar con el procesamiento que necesites

    if estilo == 'points':
        for punto in puntos:
            x, y = int(punto[0]), int(punto[1])
            cv2.drawMarker(img, (x, y), color=color_bgr, markerType=cv2.MARKER_CROSS, markerSize=10,
                           thickness=thickness)


    elif estilo == 'rectangles':

        num_puntos = puntos.shape[0]


        for i in range(num_puntos):
            x1, y1 = puntos[i][0]
            x2, y2 = puntos[(i + 1) % num_puntos][0]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))

            # Dibuja el rectángulo
            cv2.rectangle(img, pt1, pt2, color_bgr, thickness)
            if i == 0:
                # Define el ID del rectángulo y el tamaño del texto
                id_text = str(idx)
                font_scale = 0.7
                font_thickness = 2

                # Calcula el tamaño del texto
                text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                # Asumiendo que tienes las variables `text_size`, `pt1`, y `img`
                height, width, _ = img.shape  # Dimensiones de la imagen

                # Margen de seguridad para el texto
                text_margin = 5

                # Asegúrate de que el texto no se salga de la imagen
                # Verifica el borde izquierdo
                if pt1[0] < 0:
                    text_x = text_margin
                elif pt1[0] + text_size[0] > width:
                    text_x = width - text_size[0] - text_margin
                else:
                    text_x = pt1[0]

                # Verifica el borde superior
                if pt1[1] - text_size[1] - text_margin < 0:
                    text_y = text_size[1] + text_margin
                elif pt1[1] > height:
                    text_y = height - text_margin
                else:
                    text_y = pt1[1]

                # Ajusta la ubicación del rectángulo de fondo para el texto
                text_bg_tl = (text_x, text_y - text_size[1] - text_margin)
                text_bg_br = (text_x + text_size[0] + text_margin, text_y)

                # Asegúrate de que el rectángulo de fondo tampoco se salga de la imagen
                # Esto puede requerir un ajuste adicional si estás muy cerca de las esquinas

                # Dibuja el rectángulo de fondo para el texto
                cv2.rectangle(img, text_bg_tl, text_bg_br, color_bgr, cv2.FILLED)

                # Dibuja el texto sobre el rectángulo de fondo
                cv2.putText(img, id_text, (text_x + text_margin, text_y - text_margin), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), font_thickness)


    elif estilo == 'polyline':
        pts = np.array(puntos, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness)

    return puntos, color_bgr  # Devolvemos los puntos y el color que se usó finalmente.


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
                frame1 = int(row['Frame Video 1'])
                frame2 = int(row['Frame Video 2'])
            except ValueError:
                # Manejar el caso donde el valor no se puede convertir a int
                continue

            # Solo añadir a la lista si el frame existe en los diccionarios
            if frame1 in frame_to_file1 and frame2 in frame_to_file2:
                img_files1.append(frame_to_file1[frame1])
                img_files2.append(frame_to_file2[frame2])


    return img_files1, img_files2

def compare_and_filter(resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada, score_threshold=0.05):
    # Ensure both lists have the same length
    assert len(resultados_a) == len(resultados_b)

    # Initialize a list to keep track of row indices to remove
    indices_to_remove = []

    # Iterate over both lists at the same time and compare
    for i in range(len(resultados_a)):
        res_a = resultados_a[i]
        res_b = resultados_b[i]
        # Compare the scores
        if res_a['Score'] is not None and res_b['Score'] is not None:
            if res_a['Score'] < res_b['Score']:
                # Update the information in res_a with the information from res_b
                resultados_a[i] = res_b.copy()
            elif res_a['Score'] > res_b['Score']:
                # Update the information in res_b with the information from res_a
                resultados_b[i] = res_a.copy()

        # Add the index to the list of indices to remove if the score is less than the threshold
        if (res_a['Score'] is not None and res_a['Score'] < score_threshold) or \
           (res_b['Score'] is not None and res_b['Score'] < score_threshold):
            indices_to_remove.append(i)

    # Remove rows with low score from both lists, starting from the last index towards the first to not alter the indices
    for index in sorted(indices_to_remove, reverse=True):
        del resultados_a[index]
        del resultados_b[index]
        del lista_a_filtrada[index]
        del lista_b_filtrada[index]

    return resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada


def compare_and_filter_by_class(resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada, selected_class=None):
    # Asegúrate de que ambas listas tengan la misma longitud
    assert len(resultados_a) == len(resultados_b)

    # Inicializa una lista para realizar un seguimiento de los índices de las filas a eliminar
    indices_to_remove = []

    # Itera sobre ambas listas al mismo tiempo y compara
    for i in range(len(resultados_a)):
        res_a = resultados_a[i]
        res_b = resultados_b[i]

        # Filtra por la clase seleccionada, si se especifica una
        if selected_class is not None and selected_class != 'all':
            if res_a['Clase'] != selected_class:
                indices_to_remove.append(i)
                continue  # Continúa con la siguiente iteración sin hacer más comprobaciones
            if res_b['Clase'] != selected_class:
                indices_to_remove.append(i)
                continue  # Continúa con la siguiente iteración sin hacer más comprobaciones


    # Elimina filas que no coincidan con la clase seleccionada de ambas listas,
    # comenzando desde el último índice hacia el primero para no alterar los índices
    for index in sorted(indices_to_remove, reverse=True):
        del resultados_a[index]
        del resultados_b[index]
        del lista_a_filtrada[index]
        del lista_b_filtrada[index]

    return resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada


def process_csv_files(csv_paths, color_map_path, img_files, filtered_lists):
    # Load the color map JSON
    with open(color_map_path, 'r') as f:
        color_dict = json.load(f)

    # Combine both dataframes into a dictionary with a composite key and including score
    class_dict = {}
    for source, csv_path in zip(['a', 'b'], csv_paths):
        df = pd.read_csv(csv_path)
        for index, row in df.iterrows():
            key = (source, str(row['Frame']).zfill(6), row['Recorte'].split('_')[1].split('.')[0])
            class_dict[key] = (row['Género'], row['Score Género'])

    # Combine both lists with tags
    recortes_combined = []
    for source, img_file, filtered_list in zip(['a', 'b'], img_files, filtered_lists):
        frame_num = img_file.split('.')[0]
        recortes_combined.extend([
            (source, frame_num, str(recorte)) for recorte in filtered_list
        ])

    # Process the combined lists in one loop
    resultados = []
    for source, frame_num, recorte in recortes_combined:
        print(frame_num)
        print(recorte)
        clase_score_tuple = class_dict.get((source, frame_num, recorte), ("Clase no encontrada", None))
        clase, score = clase_score_tuple
        color = color_dict.get(clase, "Color no encontrado")
        print(clase)
        print(color)
        resultados.append({
            "Source": source,
            "Frame": frame_num,
            "Recorte": recorte,
            "Clase": clase,
            "Score": score,
            "Color": color
        })

    # Separate the results based on the source
    resultados_a = [result for result in resultados if result['Source'] == 'a']
    resultados_b = [result for result in resultados if result['Source'] == 'b']

    return resultados_a, resultados_b

def extract_crops_lists(csv_path1, csv_path2, idx, column_index=1):
    # Open the CSV files for reading
    with open(csv_path1, 'r', newline='', encoding='utf-8') as csvfile_a, \
            open(csv_path2, 'r', newline='', encoding='utf-8') as csvfile_b:
        # Create CSV readers for both files
        csv_reader_a = csv.reader(csvfile_a)
        csv_reader_b = csv.reader(csvfile_b)

        # Skip the header if present
        header_a = next(csv_reader_a, None)
        header_b = next(csv_reader_b, None)

        # Initialize the lists
        lista_a_filtrada = []
        lista_b_filtrada = []

        # Iterate over rows of both CSV files simultaneously using zip and enumerate
        for i, (row1, row2) in enumerate(zip(csv_reader_a, csv_reader_b)):
            if i == idx:
                # Extract the numbers from the "lista_crops" column using eval()
                lista_a_filtrada = eval(row1[column_index])
                lista_b_filtrada = eval(row2[column_index])
                break  # Exit after the required row is processed

    return lista_a_filtrada, lista_b_filtrada


def create_legend_info(resultados):
    clases_colores = {}
    for resultado in resultados:
        clase = resultado["Clase"]
        color = resultado["Color"]

        # Ignorar los resultados donde el color no es válido
        if color == 'Color no encontrado' or color is None:
            continue

        clases_colores[clase] = color

    # Crear los handles para la leyenda
    handles = [mpatches.Patch(color=np.array(color), label=clase) for clase, color in clases_colores.items()]
    return clases_colores, handles
def guardar_subplot(figura, nombre_archivo):
    ruta_guardado = os.path.join(output_root, nombre_archivo)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    figura.savefig(ruta_guardado, dpi=100, bbox_inches='tight')

def main(img_dir1, img_dir2, annotations_dir1, annotations_dir2, csv_file, csv1_path, csv2_path, csv_paths, color_map_path, output_root, selected_class):

    # Assuming that 'Frame Video 1' and 'Frame Video 2' are the column headers in the CSV
    img_files1, img_files2 = build_image_lists_from_csv(csv_file, img_dir1, img_dir2)



    for idx, (img_file1, img_file2) in enumerate(zip(img_files1, img_files2)):
        img_file12 = img_files1[idx+1]
        img_file21 = img_files2[idx+1]

        img_path1 = os.path.join(img_dir1, img_file1)
        img_path2 = os.path.join(img_dir2, img_file2)
        img_path12 = os.path.join(img_dir1, img_file12)
        img_path21 = os.path.join(img_dir2, img_file21)


        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        img12 = cv2.imread(img_path12)
        img21 = cv2.imread(img_path21)



        img1_annotations = os.path.join(annotations_dir1, img_file1.replace('.jpg', '_semantic.json'))
        img2_annotations = os.path.join(annotations_dir2, img_file2.replace('.jpg', '_semantic.json'))
        img12_annotations = os.path.join(annotations_dir1, img_file12.replace('.jpg', '_semantic.json'))
        img21_annotations = os.path.join(annotations_dir2, img_file21.replace('.jpg', '_semantic.json'))

        if os.path.isfile(img1_annotations) and os.path.isfile(img2_annotations):

            # Usar idx para cargar el mapa correcto
            map_img1 = f'/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/GX010042/mapa_{idx}.png'
            map_img2 = f'/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/maps/video1/mapa_{idx}.png'
            #map_img1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Maps/GX010042-gps-original-1.png'
            #map_img2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Maps/video1-gps-original-1.png'

            # Usage
            column_index = 1  # Assuming the "lista_crops" is in the second column

            # Call the function
            lista_a_filtrada, lista_b_filtrada = extract_crops_lists(csv1_path, csv2_path, idx, column_index)

            # Usage:
            img_files = [img_file1, img_file2]  # Assume these are defined above
            filtered_lists = [lista_a_filtrada, lista_b_filtrada]  # Assume these are defined above

            resultados_a, resultados_b = process_csv_files(csv_paths, color_map_path, img_files, filtered_lists)


            score_threshold = 0.05
            resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada = compare_and_filter(
                resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada, score_threshold
            )

            #selected_class = 'todas'  # Reemplazar con la clase deseada o 'todas' para no filtrar por clase
            resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada = compare_and_filter_by_class(
                resultados_a, resultados_b, lista_a_filtrada, lista_b_filtrada, selected_class
            )

            points_img1_list = []
            points_img2_list = []

            print(img2_annotations)
            print(lista_b_filtrada)
            original_points_img1 = transformar_puntos_bbox(img1_annotations, homography_matrix=None, transformar=False,
                                                           img_matchings=lista_a_filtrada)
            original_points_img2 = transformar_puntos_bbox(img2_annotations, homography_matrix=None, transformar=False,
                                                           img_matchings=lista_b_filtrada)

            for i in range(len(original_points_img1)):
                points_img1_list.append((original_points_img1[i], resultados_a[i]["Color"]))
                points_img2_list.append((original_points_img2[i], resultados_b[i]["Color"]))
                #print(resultados_a[i]["Clase"], resultados_b[i]["Clase"])



            img1_copy = img1.copy()
            img2_copy = img2.copy()
            img12_copy = img12.copy()



            for i, (puntos_y_color_img1, puntos_y_color_img2) in enumerate(zip(points_img1_list, points_img2_list)):
                dibujar_puntos1(img1_copy, puntos_y_color_img1, i, estilo='rectangles', thickness=2)
                dibujar_puntos1(img2_copy, puntos_y_color_img2, i, estilo='rectangles', thickness=2)

            # Process the results for both lists and get clases_colores and handles
            clases_colores_a, handles_a = create_legend_info(resultados_a)
            clases_colores_b, handles_b = create_legend_info(resultados_b)

            plt.figure(figsize=(24, 16))  # Aumenta el tamaño de la figura para acomodar los 4 subgráficos

            # Subgráfico 1: Mapa 1
            plt.subplot(2, 2, 1)
            plt.imshow(plt.imread(map_img1))
            # plt.title('Map 1')
            plt.axis('off')

            # Subgráfico 2: Imagen 1
            plt.subplot(2, 2, 3)
            plt.imshow(cv2.cvtColor(img1_copy, cv2.COLOR_BGR2RGB))
            # plt.title('Image 1')
            plt.axis('off')
            plt.legend(handles=handles_a, loc='lower right')

            # Subgráfico 3: Mapa 2
            plt.subplot(2, 2, 2)
            plt.imshow(plt.imread(map_img2))
            # plt.title('Map 2')
            plt.axis('off')

            # Subgráfico 4: Imagen 2
            plt.subplot(2, 2, 4)
            plt.imshow(cv2.cvtColor(img2_copy, cv2.COLOR_BGR2RGB))
            # plt.title('Image 2')
            plt.axis('off')
            plt.legend(handles=handles_b, loc='lower right')

            # Ajustar la disposición de los subgráficos
            plt.subplots_adjust(top=0.1, bottom=0.01, left=0.01, right=0.10, hspace=0.25, wspace=0.01)

            plt.tight_layout()






            base_name = os.path.splitext(os.path.basename(img1_annotations))[0]
            # Construye la ruta completa donde se guardará la imagen
            ruta_guardado = os.path.join(output_root, f"{base_name}_{selected_class}_bbox.jpg")

            # Verifica si el directorio existe y, si no, crea el directorio
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            # Asumiendo que ya has creado una figura con plt y quieres guardarla
            plt.savefig(ruta_guardado, dpi=100, bbox_inches='tight')
            '''plt.show(block=False)
            plt.pause(3)
            plt.close()'''
            print(f"Bounding boxes guardadas en {ruta_guardado} para {base_name}.json")

            # Subgráfico 1: Mapa 1
            fig1 = plt.figure(figsize=(6, 4))
            plt.imshow(plt.imread(map_img1))
            plt.axis('off')
            guardar_subplot(fig1, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Reidentification/map1.jpg')

            # Subgráfico 2: Imagen 1
            fig2 = plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(img1_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.legend(handles=handles_a, loc='lower right')
            guardar_subplot(fig2, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Reidentification/imagen1.jpg')

            # Subgráfico 3: Mapa 2
            fig3 = plt.figure(figsize=(6, 4))
            plt.imshow(plt.imread(map_img2))
            plt.axis('off')
            guardar_subplot(fig3, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Reidentification/map2.jpg')

            # Subgráfico 4: Imagen 2
            fig4 = plt.figure(figsize=(6, 4))
            plt.imshow(cv2.cvtColor(img2_copy, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.legend(handles=handles_b, loc='lower right')
            guardar_subplot(fig4, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Reidentification/imagen2.jpg')

        else:
            print(f"Falta el archivo de anotaciones para {img_file1} o {img_file2}.")



if __name__ == "__main__":

    video1_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
    video2_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img'
    video1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation'
    video2_annotations = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation'
    matched_frames_csv = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/matched_frames/matched_frames_euclidean_corrected.csv'
    matching1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_GX010042.csv'
    matching2_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_video1.csv'
    class_score_csv = [
        "/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Resultados/GX010042_classes.csv",
        "/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/Resultados/video1_classes.csv"
    ]
    color_map_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/PlantNet-API/mapa_de_colores_aux.json'
    selected_class = 'all'
    output_root = f"/mnt/rhome/rlg/PROYECTO/Homography/visualizador_combined_aux/{selected_class}"



    main(video1_path, video2_path, video1_annotations, video2_annotations, matched_frames_csv,
         matching1_annotations, matching2_annotations, class_score_csv, color_map_path, output_root,
         selected_class)