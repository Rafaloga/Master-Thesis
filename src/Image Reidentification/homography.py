import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt


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


def dibujar_bounding_boxes(img_annotations_path, homography_image, color):
    with open(img_annotations_path, 'r') as img_annotations_file:
        data = json.load(img_annotations_file)

    annotations = data["annotations"]

    '''if annotations:
        annotation = annotations[8]  # Obtener la primera anotación
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox
        cv2.rectangle(homography_image, (x_min, y_min), (x_min + width, y_min + height), color, 2)
    '''
    for annotation in annotations:
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox
        cv2.rectangle(homography_image, (x_min, y_min), (x_min + width, y_min + height), color, 2)


def dibujar_bounding_boxes_transformada(img_annotations_path, homography_image, homography_matrix, color):
    with open(img_annotations_path, 'r') as img_annotations_file:
        data = json.load(img_annotations_file)

    annotations = data["annotations"]

    '''if annotations:
        annotation = annotations[0]  # Obtener la primera anotación
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox

        # Aplicar la transformación de la matriz de homografía a las coordenadas de la bounding box
        pts = np.array(
            [[x_min, y_min], [x_min + width, y_min], [x_min + width, y_min + height], [x_min, y_min + height]],
            dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)


        # Usar cv2.perspectiveTransform para aplicar la transformación
        transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)

        # Dibujar marcadores en cada punto transformado
        for point in transformed_pts:
            x, y = int(point[0][0]), int(point[0][1])
            cv2.drawMarker(homography_image, (x, y), color=color, markerType=cv2.MARKER_CROSS, markerSize=10,
                           thickness=2)
        '''





    for annotation in annotations:
        bbox = annotation["bbox"]
        x_min, y_min, width, height = bbox

        # Aplicar la transformación de la matriz de homografía a las coordenadas de la bounding box
        pts = np.array(
            [[x_min, y_min], [x_min + width, y_min], [x_min + width, y_min + height], [x_min, y_min + height]],
            dtype=np.float32)
        pts = pts.reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)

        # Dibujar la bounding box transformada en la imagen
        #cv2.polylines(homography_image, [np.int32(transformed_pts)], isClosed=True, color=color, thickness=2)
        # Dibujar marcadores en cada punto transformado
        for point in transformed_pts:
            x, y = int(point[0][0]), int(point[0][1])
            cv2.drawMarker(homography_image, (x, y), color=color, markerType=cv2.MARKER_CROSS, markerSize=10,
                           thickness=2)

def main():
    # Rutas de las imágenes y archivos JSON
    #img1 = cv2.imread('/mnt/rhome/rlg/PROYECTO/Homography/image/example1.png')
    #img2 = cv2.imread('/mnt/rhome/rlg/PROYECTO/Homography/image/example2.png')
    #img1_path = '/mnt/rhome/rlg/PROYECTO/Homography/image/example1.png'
    #img2_path = '/mnt/rhome/rlg/PROYECTO/Homography/image/example2.png'

    img1 = cv2.imread('/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img/000000.jpg')
    img2 = cv2.imread('/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/000000.jpg')
    img1_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/img/000000.jpg'
    img2_path = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/img/000000.jpg'
    img1_annotations = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation/000000_semantic.json'
    img2_annotations = '/mnt/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation/000000_semantic.json'

    # Calcular la homografía (img1 se transforma a la perspectiva de img2)
    homography_image, homography_matrix = calcular_homografia(img2_path, img1_path)



    if homography_image is not None:
        # Dibujar bounding boxes en la imagen transformada
        #dibujar_bounding_boxes_transformada(img1_annotations, img2, homography_matrix, (0, 255, 0))
        dibujar_bounding_boxes_transformada(img2_annotations, img1, homography_matrix, (0, 0, 255))
        dibujar_bounding_boxes(img1_annotations, img1, (0, 255, 0))
        # Mostrar la imagen resultante
        plt.imshow(img1)
        plt.show()

        # Guardar la imagen resultante
        output_root = '/mnt/rhome/rlg/PROYECTO/Homography/homography'
        base_name = os.path.splitext(os.path.basename(img1_annotations))[0]
        output_path = os.path.join(output_root, base_name.replace('_semantic', ''))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        ruta_guardado = os.path.join(output_path, f"{base_name}_bbox.jpg")
        cv2.imwrite(ruta_guardado, img1)
        print(f"Bounding boxes guardadas en {ruta_guardado} para {base_name}.json")
    else:
        print("No se encontraron suficientes coincidencias para calcular la homografía.")


if __name__ == "__main__":
    main()
