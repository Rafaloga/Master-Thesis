import os
import requests
import json
import sys
import csv
import re
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# Creamos una nueva función que procesa un conjunto de datos completo

def calculate_iou_matrix(annotations1, annotations2):
    iou_matrix = np.zeros((len(annotations1), len(annotations2)))

    for i, annotation1 in enumerate(annotations1):
        for j, annotation2 in enumerate(annotations2):
            iou = calculate_iou(annotation1, annotation2)
            iou_matrix[i][j] = iou

    return iou_matrix
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


def leer_matching_dict(matching_dict):
    img1_matchings = []
    img2_matchings = []

    for img1_match, img2_match in matching_dict.items():
        img1_matchings.append(int(img1_match))
        img2_matchings.append(int(img2_match))

    return img1_matchings, img2_matchings


def create_csv(img_dir1, img_dir2, annotations_dir1, annotations_dir2, json1_salida, json2_salida):
    img_files1 = sorted(os.listdir(img_dir1))
    img_files2 = sorted(os.listdir(img_dir2))


    with open(json1_salida, 'w', newline='', encoding='utf-8') as csvfile_a, \
            open(json2_salida, 'w', newline='', encoding='utf-8') as csvfile_b:
        writer_a = csv.writer(csvfile_a)
        writer_b = csv.writer(csvfile_b)
        # Escribir los encabezados si lo deseas
        writer_a.writerow(['img', 'lista_crops'])
        writer_b.writerow(['img', 'lista_crops'])

    for idx, (img_file1, img_file2) in enumerate(zip(img_files1, img_files2)):

        img_file12 = img_files1[idx + 1]
        img_file21 = img_files2[idx + 1]
        img_path1 = os.path.join(img_dir1, img_file1)
        img_path2 = os.path.join(img_dir2, img_file2)
        img_path12 = os.path.join(img_dir1, img_file12)
        img_path21 = os.path.join(img_dir2, img_file21)

        img1_annotations = os.path.join(annotations_dir1, img_file1.replace('.jpg', '_semantic.json'))
        img2_annotations = os.path.join(annotations_dir2, img_file2.replace('.jpg', '_semantic.json'))
        img12_annotations = os.path.join(annotations_dir1, img_file12.replace('.jpg', '_semantic.json'))
        img21_annotations = os.path.join(annotations_dir2, img_file21.replace('.jpg', '_semantic.json'))

        if os.path.isfile(img1_annotations) and os.path.isfile(img2_annotations):
            homography_image, homography_matrix = calcular_homografia(img_path2, img_path1)
            homography_image12, homography_matrix12 = calcular_homografia(img_path12, img_path1)
            homography_image21, homography_matrix21 = calcular_homografia(img_path21, img_path2)

            if homography_image is not None:

                original_points = transformar_puntos_bbox_wo(img1_annotations, homography_matrix, transformar=False)
                transformed_points = transformar_puntos_bbox_wo(img2_annotations, homography_matrix, transformar=True)

                original_points12 = transformar_puntos_bbox_wo(img1_annotations, homography_matrix12, transformar=False)
                transformed_points12 = transformar_puntos_bbox_wo(img12_annotations, homography_matrix12,
                                                                  transformar=True)

                original_points21 = transformar_puntos_bbox_wo(img2_annotations, homography_matrix21, transformar=False)
                transformed_points21 = transformar_puntos_bbox_wo(img21_annotations, homography_matrix21,
                                                                  transformar=True)

                iou_threshold = 0.4

                matching_dict, matching_original_points, matching_transformed_points = find_matching_annotations_wo(
                    original_points,
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

                with open(json1_salida, 'a', newline='', encoding='utf-8') as csvfile_a, \
                        open(json2_salida, 'a', newline='', encoding='utf-8') as csvfile_b:
                    writer_a = csv.writer(csvfile_a)
                    writer_b = csv.writer(csvfile_b)
                    img_pair_1 = f"{img_file1}"
                    img_pair_2 = f"{img_file2}"
                    writer_a.writerow([img_pair_1, lista_a_filtrada])
                    writer_b.writerow([img_pair_2, lista_b_filtrada])




video1_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img'
video2_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img'
video1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation'
video2_annotations = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation'
csv1_salida = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_GX010042.csv'
csv2_salida = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv'
create_csv(video1_path, video2_path, video1_annotations, video2_annotations, csv1_salida, csv2_salida)







