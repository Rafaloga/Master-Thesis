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
def calcular_homografia(img1_path, img2_path):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = cv2.warpPerspective(img1, M, (img1.shape[1], img1.shape[0]))
        return result, M
    else:
        return None, None


def transformar_puntos_bbox(img_annotations_path, homography_matrix, transformar=True, img_matchings=None):

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







def calculate_iou(annotation1, annotation2):
    annotation1 = np.array(annotation1).reshape(4, 2)
    annotation2 = np.array(annotation2).reshape(4, 2)

    x1, y1 = annotation1[:, 0], annotation1[:, 1]
    x2, y2 = annotation2[:, 0], annotation2[:, 1]

    x_overlap = max(0, min(x1[2], x2[2]) - max(x1[0], x2[0]))
    y_overlap = max(0, min(y1[2], y2[2]) - max(y1[0], y2[0]))

    intersection = x_overlap * y_overlap
    union = ((x1[2] - x1[0]) * (y1[2] - y1[0])) + ((x2[2] - x2[0]) * (y2[2] - y2[0])) - intersection

    iou = intersection / union
    return iou


def calculate_iou_matrix(annotations1, annotations2):
    iou_matrix = np.zeros((len(annotations1), len(annotations2)))

    for i, annotation1 in enumerate(annotations1):
        for j, annotation2 in enumerate(annotations2):
            iou = calculate_iou(annotation1, annotation2)
            iou_matrix[i][j] = iou

    return iou_matrix


def find_matching_annotations(annotations1, annotations2, iou_threshold):
    iou_matrix = calculate_iou_matrix(annotations1, annotations2)

    # Encuentra las asignaciones óptimas utilizando el algoritmo Húngaro
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matching_original_points = []
    matching_transformed_points = []

    # Crear un diccionario para mapear las anotaciones originales a las transformadas
    matching_dict = {}
    for i, j in zip(row_ind, col_ind):
        iou_value = iou_matrix[i][j]
        if iou_value >= iou_threshold:
            matching_original_points.append(annotations1[i])
            matching_transformed_points.append(annotations2[j])
            matching_dict[int(i)] = int(j)  # Convertir claves a enteros

    # Guardar el diccionario en un archivo JSON
    matching_dict_file_path = 'matching_dict.json'
    with open(matching_dict_file_path, 'w') as matching_dict_file:
        json.dump(matching_dict, matching_dict_file)

    return matching_original_points, matching_transformed_points

def leer_matching_dict(matching_dict):
    img1_matchings = []
    img2_matchings = []

    for img1_match, img2_match in matching_dict.items():
        img1_matchings.append(int(img1_match))
        img2_matchings.append(int(img2_match))


    return img1_matchings, img2_matchings

def dibujar_puntos(img, points, style='points', thickness=2):
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 0, 255)]  # Puedes agregar más colores si es necesario
    color_idx = 0  # Inicializa el índice de color

    for pts in points:
        color = colors[color_idx]  # Obtiene el color actual
        color_idx = (color_idx + 1) % len(colors)  # Avanza al siguiente color

        for point in pts:
            x, y = int(point[0][0]), int(point[0][1])
            if style == 'points':
                cv2.drawMarker(img, (x, y), color=color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=thickness)
            elif style == 'rectangles':
                for i in range(len(pts) - 1):
                    cv2.rectangle(img, (int(pts[i][0][0]), int(pts[i][0][1])),
                                  (int(pts[i + 1][0][0]), int(pts[i + 1][0][1])), color, thickness)

                    cv2.rectangle(img, (int(pts[-1][0][0]), int(pts[-1][0][1])),
                                  (int(pts[0][0][0]), int(pts[0][0][1])), color, thickness)
            elif style == 'polyline':
                cv2.polylines(img, [np.int32(pts)], isClosed=True, color=color, thickness=thickness)

def mostrar_mapa_desde_csv(csv_path):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(csv_path)

    # Extraer las coordenadas latitud y longitud
    latitudes = df['lat'].tolist()
    longitudes = df['long'].tolist()

    # Crear un mapa básico
    plt.figure(figsize=(10, 10))
    m = Basemap(projection='merc', resolution='l', llcrnrlat=min(latitudes), urcrnrlat=max(latitudes), llcrnrlon=min(longitudes), urcrnrlon=max(longitudes))

    # Dibujar la costa y las fronteras del mapa
    m.drawcoastlines()
    m.drawcountries()

    # Convertir las coordenadas a las coordenadas del mapa
    x, y = m(longitudes, latitudes)

    # Dibujar los puntos en el mapa
    m.scatter(x, y, marker='o', color='g', s=10, zorder=5)

    plt.savefig('mapa_video1.png')
    # Mostrar el mapa
    plt.show()

def transformar_puntos_bbox_wo(img_annotations_path, homography_matrix, transformar=True):
    with open(img_annotations_path, 'r') as img_annotations_file:
        data = json.load(img_annotations_file)

    annotations = data["annotations"]
    transformed_points = []

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

def find_matching_annotations_wo(annotations1, annotations2, iou_threshold):
    iou_matrix = calculate_iou_matrix(annotations1, annotations2)

    # Encuentra las asignaciones óptimas utilizando el algoritmo Húngaro
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    matching_original_points = []
    matching_transformed_points = []

    # Crear un diccionario para mapear las anotaciones originales a las transformadas
    matching_dict = {}
    for i, j in zip(row_ind, col_ind):
        iou_value = iou_matrix[i][j]
        if iou_value >= iou_threshold:
            matching_original_points.append(annotations1[i])
            matching_transformed_points.append(annotations2[j])
            matching_dict[int(i)] = int(j)  # Convertir claves a enteros


    return matching_dict, matching_original_points, matching_transformed_points


def find_matching_annotations_4(annotations1, annotations2, annotations3, annotations4, iou_threshold):
    iou_matrix1_2 = calculate_iou_matrix(annotations1, annotations2)
    iou_matrix1_3 = calculate_iou_matrix(annotations1, annotations3)
    iou_matrix1_4 = calculate_iou_matrix(annotations1, annotations4)

    # Calcular la matriz de IoU entre annotations2 y annotations3
    iou_matrix2_3 = calculate_iou_matrix(annotations2, annotations3)

    # Calcular la matriz de IoU entre annotations2 y annotations4
    iou_matrix2_4 = calculate_iou_matrix(annotations2, annotations4)

    # Calcular la matriz de IoU entre annotations3 y annotations4
    iou_matrix3_4 = calculate_iou_matrix(annotations3, annotations4)

    # Encuentra las asignaciones óptimas utilizando el algoritmo Húngaro para cada par de matrices IoU
    row_ind_1_2, col_ind_1_2 = linear_sum_assignment(-iou_matrix1_2)
    row_ind_1_3, col_ind_1_3 = linear_sum_assignment(-iou_matrix1_3)
    row_ind_1_4, col_ind_1_4 = linear_sum_assignment(-iou_matrix1_4)
    row_ind_2_3, col_ind_2_3 = linear_sum_assignment(-iou_matrix2_3)
    row_ind_2_4, col_ind_2_4 = linear_sum_assignment(-iou_matrix2_4)
    row_ind_3_4, col_ind_3_4 = linear_sum_assignment(-iou_matrix3_4)

    matching_original_points = []
    matching_transformed_points = []

    # Crear un diccionario para mapear las anotaciones originales a las transformadas para cada par de matrices IoU
    matching_dict_1_2 = {}
    for i, j in zip(row_ind_1_2, col_ind_1_2):
        iou_value = iou_matrix1_2[i][j]
        if iou_value >= iou_threshold:
            matching_original_points.append(annotations1[i])
            matching_transformed_points.append(annotations2[j])
            matching_dict_1_2[int(i)] = int(j)  # Convertir claves a enteros

    matching_dict_1_3 = {}
    for i, j in zip(row_ind_1_3, col_ind_1_3):
        iou_value = iou_matrix1_3[i][j]
        if iou_value >= iou_threshold:
            matching_original_points.append(annotations1[i])
            matching_transformed_points.append(annotations3[j])
            matching_dict_1_3[int(i)] = int(j)  # Convertir claves a enteros

    # Repite el proceso para los otros pares de matrices IoU (2_3, 2_4 y 3_4) si es necesario.

    return (matching_dict_1_2, matching_dict_1_3), matching_original_points, matching_transformed_points


def calcular_iou(rect1, rect2):
    # Calcula la intersección de los rectángulos
    xA = max(rect1[0], rect2[0])
    yA = max(rect1[1], rect2[1])
    xB = min(rect1[2], rect2[2])
    yB = min(rect1[3], rect2[3])

    # Calcula el área de intersección
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Calcula el área de cada rectángulo
    boxAArea = (rect1[2] - rect1[0] + 1) * (rect1[3] - rect1[1] + 1)
    boxBArea = (rect2[2] - rect2[0] + 1) * (rect2[3] - rect2[1] + 1)

    # Calcula la unión de los rectángulos
    unionArea = boxAArea + boxBArea - interArea

    # Calcula el IoU
    iou = interArea / float(unionArea)

    return iou



def dibujar_puntos1(img, puntos_y_color, puntos_y_color_list_prev=None, estilo='points', thickness=2,
                    iou_threshold=0.5):
    puntos, color = puntos_y_color
    color_bgr = tuple(reversed(color))
    new_color = color_bgr  # Color por defecto

    if puntos_y_color_list_prev:
        iou_max = 0

        for puntos_prev, color_prev in puntos_y_color_list_prev:
            for i in range(len(puntos)):
                rect1 = (puntos[i][0][0], puntos[i][0][1], puntos[(i + 1) % len(puntos)][0][0],
                         puntos[(i + 1) % len(puntos)][0][1])
                rect2 = (puntos_prev[i][0][0], puntos_prev[i][0][1], puntos_prev[(i + 1) % len(puntos_prev)][0][0],
                         puntos_prev[(i + 1) % len(puntos_prev)][0][1])

                iou = calcular_iou(rect1, rect2)

                if iou > iou_max:
                    iou_max = iou
                    new_color = tuple(reversed(color_prev))

        if iou_max > iou_threshold:
            color_bgr = new_color

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

            cv2.rectangle(img, pt1, pt2, color_bgr, thickness)

    elif estilo == 'polyline':
        pts = np.array(puntos, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color_bgr, thickness=thickness)

    return puntos, color_bgr  # Devolvemos los puntos y el color que se usó finalmente.


def main(img_dir1, img_dir2, annotations_dir1, annotations_dir2):
    img_files1 = sorted(os.listdir(img_dir1))
    img_files2 = sorted(os.listdir(img_dir2))
    img_files1 = img_files1[0:100]
    img_files2 = img_files2[0:100]

    img1_list = []
    img2_list = []
    puntos_y_color_actualizados1 = []
    puntos_y_color_actualizados2 = []
    img1_list_actualizada = []
    img2_list_actualizada = []
    match_img1 = []
    match_img2 = []
    list1 = []
    list2 = []
    nueva_lista = []
    nueva_lista2 = []

    with open('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_GX010042.csv', 'w', newline='', encoding='utf-8') as csvfile_a, \
            open('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv', 'w', newline='', encoding='utf-8') as csvfile_b:
        writer_a = csv.writer(csvfile_a)
        writer_b = csv.writer(csvfile_b)
        # Escribir los encabezados si lo deseas
        writer_a.writerow(['img', 'lista_crops'])
        writer_b.writerow(['img', 'lista_crops'])

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
            homography_image, homography_matrix = calcular_homografia(img_path2, img_path1)
            homography_image12, homography_matrix12 = calcular_homografia(img_path12, img_path1)
            homography_image21, homography_matrix21 = calcular_homografia(img_path21, img_path2)

            # Usar idx para cargar el mapa correcto
            map_img1 = f'/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/maps/GX010042/mapa_{idx}.png'
            map_img2 = f'/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/maps/video1/mapa_{idx}.png'
            if homography_image is not None:

                original_points = transformar_puntos_bbox_wo(img1_annotations, homography_matrix, transformar=False)
                transformed_points = transformar_puntos_bbox_wo(img2_annotations, homography_matrix, transformar=True)

                original_points12 = transformar_puntos_bbox_wo(img1_annotations, homography_matrix12, transformar=False)
                transformed_points12 = transformar_puntos_bbox_wo(img12_annotations, homography_matrix12, transformar=True)

                original_points21 = transformar_puntos_bbox_wo(img2_annotations, homography_matrix21, transformar=False)
                transformed_points21 = transformar_puntos_bbox_wo(img21_annotations, homography_matrix21, transformar=True)

                iou_threshold = 0.4

                matching_dict, matching_original_points, matching_transformed_points = find_matching_annotations_wo(original_points,
                                                                                                  transformed_points,
                                                                                                  iou_threshold)
                matching_dict12, matching_original_points12, matching_transformed_points12 = find_matching_annotations_wo(
                    original_points12,
                    transformed_points12,
                    iou_threshold)
                matching_dict21, matching_original_points21, matching_transformed_points21 = find_matching_annotations_wo(
                    original_points21,
                    transformed_points21,
                    iou_threshold)

                img1_matchings, img2_matchings = leer_matching_dict(matching_dict)
                img1_matchings12, img2_matchings12 = leer_matching_dict(matching_dict12)
                img1_matchings21, img2_matchings21 = leer_matching_dict(matching_dict21)
                list1.append(img1_matchings)
                list2.append(img2_matchings)
                match_img1.append(img2_matchings12)
                match_img2.append(img2_matchings21)
                #print(img1_matchings)
                #print(img2_matchings)
                #print(img1_matchings12)
                #print(img2_matchings12)


                conjunto1 = set(img1_matchings12)
                resultado1 = [(indice1, valor1) for indice1, valor1 in enumerate(img1_matchings) if valor1 in conjunto1]

                conjunto2 = set(img1_matchings21)
                resultado2 = [(indice2, valor2) for indice2, valor2 in enumerate(img2_matchings) if valor2 in conjunto2]

                # Crear diccionarios a partir de los resultados para una búsqueda más rápida
                dict1 = {indice: valor for indice, valor in resultado1}
                dict2 = {indice: valor for indice, valor in resultado2}


                # Encontrar el tamaño adecuado para las listas
                tamaño_lista = max(max(dict1.keys(), default=0), max(dict2.keys(), default=0)) + 1

                # Inicializar listas_a y listas_b
                lista_a = [None] * tamaño_lista
                lista_b = [None] * tamaño_lista

                # Recorrer los diccionarios y llenar las listas
                for indice in dict1:
                    if indice in dict2:
                        lista_a[indice] = dict1[indice]
                        lista_b[indice] = dict2[indice]

                lista_a_filtrada = [valor for valor in lista_a if valor is not None]
                lista_b_filtrada = [valor for valor in lista_b if valor is not None]
                resultado_a = lista_a_filtrada
                resultado_b = lista_b_filtrada
                print('Frame:', idx)
                print("Lista A filtrada:", lista_a_filtrada)
                print("Lista B filtrada:", lista_b_filtrada)

                with open('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_GX010042.csv', 'a', newline='', encoding='utf-8') as csvfile_a, \
                        open('/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv', 'a', newline='', encoding='utf-8') as csvfile_b:
                    writer_a = csv.writer(csvfile_a)
                    writer_b = csv.writer(csvfile_b)
                    img_pair_1 = f"{img_file1}"
                    img_pair_2 = f"{img_file2}"
                    writer_a.writerow([img_pair_1, lista_a_filtrada])
                    writer_b.writerow([img_pair_2, lista_b_filtrada])

                '''points_img1_list = []
                points_img2_list = []


                original_points_img1 = transformar_puntos_bbox(img1_annotations, homography_matrix, transformar=False,
                                                               img_matchings=lista_a_filtrada)
                original_points_img2 = transformar_puntos_bbox(img2_annotations, homography_matrix, transformar=False,
                                                               img_matchings=lista_b_filtrada)

                for i in range(len(original_points_img1)):
                    color = colors[i]
                    points_img1_list.append((original_points_img1[i], color))
                    points_img2_list.append((original_points_img2[i], color))

                img1_list.append(points_img1_list)
                img2_list.append(points_img2_list)

                img1_copy = img1.copy()
                img2_copy = img2.copy()
                img12_copy = img12.copy()
                img21_copy = img21.copy()

                # Lista para almacenar los puntos y colores actualizados

                #if idx < 10:
                for puntos_y_color in points_img1_list:
                    dibujar_puntos1(img1_copy, puntos_y_color, estilo='rectangles', thickness=2)
                for puntos_y_color in points_img2_list:
                    dibujar_puntos1(img2_copy, puntos_y_color, estilo='rectangles', thickness=2)
                img1_list_actualizada.append(points_img1_list)
                img2_list_actualizada.append(points_img2_list)
                else:
                    for puntos_y_color in img1_list[idx]:
                        puntos_actualizados1, color_actualizado1 = dibujar_puntos1(img1_copy, puntos_y_color,
                                                                                 img1_list[idx-1], estilo='rectangles',
                                                                                 thickness=2, iou_threshold=0.1)
                        puntos_y_color_actualizados1.append((puntos_actualizados1, color_actualizado1))
                    for puntos_y_color in img2_list[idx]:
                        puntos_actualizados2, color_actualizado2 = dibujar_puntos1(img2_copy, puntos_y_color,
                                                                                 img2_list[idx-1], estilo='rectangles',
                                                                                 thickness=2, iou_threshold=0.1)
                        puntos_y_color_actualizados2.append((puntos_actualizados2, color_actualizado2))

                    for (puntos_y_color1, puntos_y_color2) in zip(img1_list[idx], img2_list[idx]):
                        # Procesamiento para img1_list
                        puntos_actualizados1, color_actualizado1 = dibujar_puntos1(img1_copy, puntos_y_color1,
                                                                                   img1_list_actualizada[idx - 1],
                                                                                   estilo='rectangles',
                                                                                   thickness=2, iou_threshold=0.1)
                        puntos_y_color_actualizados1.append((puntos_actualizados1, color_actualizado1))

                        # Procesamiento para img2_list
                        puntos_actualizados2, color_actualizado2 = dibujar_puntos1(img2_copy, puntos_y_color2,
                                                                                   img2_list_actualizada[idx - 1],
                                                                                   estilo='rectangles',
                                                                                   thickness=2, iou_threshold=0.1)
                        puntos_y_color_actualizados2.append((puntos_actualizados2, color_actualizado1))

                    img1_list_actualizada.append(puntos_y_color_actualizados1)
                    img2_list_actualizada.append(puntos_y_color_actualizados2)
                if idx == 0:
                    dibujar_puntos(img1_copy, original_points_img1, style='rectangles')
                    dibujar_puntos(img2_copy, original_points_img2, style='rectangles')

                else:
                    dibujar_puntos1(img1_copy, original_points_img1, points_img1_list[idx - 1], style='rectangles')
                    dibujar_puntos1(img2_copy, original_points_img2, points_img2_list[idx - 1], style='rectangles')


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

                # Ajustar la disposición de los subgráficos
                plt.subplots_adjust(top=0.1, bottom=0.01, left=0.01, right=0.10, hspace=0.25, wspace=0.35)

                plt.tight_layout()



                output_root = '/mnt/rhome/rlg/PROYECTO/Homography/homography/visualizador1'
                base_name = os.path.splitext(os.path.basename(img1_annotations))[0]
                ruta_guardado = os.path.join(output_root, f"{base_name}_bbox.jpg")
                plt.savefig(ruta_guardado, orientation='landscape')
                #plt.show()
                print(f"Bounding boxes guardadas en {ruta_guardado} para {base_name}.json")
'''
            else:
                print(f"No se encontraron suficientes coincidencias para calcular la homografía para {img_file1} y {img_file2}.")
        else:
            print(f"Falta el archivo de anotaciones para {img_file1} o {img_file2}.")



if __name__ == "__main__":

    video1_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
    video2_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img'
    video1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation'
    video2_annotations = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation'



    main(video1_path, video2_path, video1_annotations, video2_annotations)