import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import pandas as pd
from mpl_toolkits.basemap import Basemap


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

def leer_matching_dict(matching_dict_file_path):
    img1_matchings = []
    img2_matchings = []

    with open(matching_dict_file_path, 'r') as matching_dict_file:
        matching_dict = json.load(matching_dict_file)

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


def main():
    img1 = cv2.imread('/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img/000000.jpg')
    img2 = cv2.imread('/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/000000.jpg')
    img1_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img/000000.jpg'
    img2_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/000000.jpg'
    img1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation/000000_semantic.json'
    img2_annotations = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation/000000_semantic.json'

    # Llamar a la función con la ruta del archivo CSV
    map_img1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/maps/GX010042/mapa_0.png'
    map_img2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/code/maps/video1/mapa_0.png'
    homography_image, homography_matrix = calcular_homografia(img2_path, img1_path)

    if homography_image is not None:
        # Llama a la función para leer el archivo matching_dict.json
        img1_matchings, img2_matchings = leer_matching_dict('matching_dict.json')

        original_points_img1 = transformar_puntos_bbox(img1_annotations, homography_matrix, transformar=False,
                                                       img_matchings=img1_matchings)
        original_points_img2 = transformar_puntos_bbox(img2_annotations, homography_matrix, transformar=False,
                                                       img_matchings=img2_matchings)

        img_wo_iou = img1.copy()
        img_w_iou = img1.copy()
        img1_copy = img1.copy()
        img2_copy = img2.copy()

        dibujar_puntos(img1_copy, original_points_img1, style='rectangles')
        dibujar_puntos(img2_copy, original_points_img2, style='rectangles')

        plt.figure(figsize=(12, 16))  # Aumenta el tamaño de la figura para acomodar los 4 subgráficos

        # Subgráfico 1: Mapa 1
        plt.subplot(2, 2, 1)
        plt.imshow(plt.imread(map_img1))
        #plt.title('Map 1')
        plt.axis('off')

        # Subgráfico 2: Imagen 1
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(img1_copy, cv2.COLOR_BGR2RGB))
        #plt.title('Image 1')
        plt.axis('off')

        # Subgráfico 3: Mapa 2
        plt.subplot(2, 2, 2)
        plt.imshow(plt.imread(map_img2))
        #plt.title('Map 2')
        plt.axis('off')

        # Subgráfico 4: Imagen 2
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(img2_copy, cv2.COLOR_BGR2RGB))
        #plt.title('Image 2')
        plt.axis('off')

        # Ajustar la disposición de los subgráficos
        plt.subplots_adjust(top=0.1, bottom=0.01, left=0.01, right=0.10, hspace=0.25, wspace=0.35)

        plt.tight_layout()
        plt.show()

        output_root = '/mnt/rhome/rlg/PROYECTO/Homography/homography'
        base_name = os.path.splitext(os.path.basename(img1_annotations))[0]
        output_path = os.path.join(output_root, base_name.replace('_semantic', ''))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        ruta_guardado = os.path.join(output_path, f"{base_name}_bbox.jpg")
        cv2.imwrite(ruta_guardado, img_w_iou)
        print(f"Bounding boxes guardadas en {ruta_guardado} para {base_name}.json")
    else:
        print("No se encontraron suficientes coincidencias para calcular la homografía.")

if __name__ == "__main__":
    main()

