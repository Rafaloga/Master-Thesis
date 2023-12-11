'''import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Cargar los datos CSV
video1 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/video1-frame_gps_interp.csv', header=0)
video2 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/GX010042-frame_gps_interp.csv', header=0)

#video1 = video1[0:100]
#video2 = video2[0:100]
# Función para calcular la distancia entre dos puntos GPS
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.deg2rad(lat1)) * np.cos(np.deg2rad(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Crear una matriz de costos basada en las distancias entre los puntos de los dos videos
num_frames_video1 = len(video1)
num_frames_video2 = len(video2)
cost_matrix = np.zeros((num_frames_video1, num_frames_video2))
for i in range(num_frames_video1):
    for j in range(num_frames_video2):
        distance = haversine(video1.iloc[i]['lat'], video1.iloc[i]['long'], video2.iloc[j]['lat'], video2.iloc[j]['long'])
        cost_matrix[i, j] = distance
        print('i, j, distance:', i, j, distance)
        
#print(cost_matrix.shape)
# Aplicar el algoritmo húngaro para encontrar los pares de frames más cercanos
row_ind, col_ind = linear_sum_assignment(cost_matrix)


# Crear matrices de coordenadas para ambos conjuntos de datos
coords_video1 = video1[['lat', 'long']].values
coords_video2 = video2[['lat', 'long']].values

# Calcular la matriz de distancias usando cdist
distances = cdist(coords_video1, coords_video2, metric='euclidean')

# Aplicar el algoritmo húngaro para encontrar los pares de frames más cercanos
row_ind, col_ind = linear_sum_assignment(distances)




# Guardar los resultados en un nuevo archivo CSV
resultados = pd.DataFrame(columns=['Frame Video 1', 'Lat Video 1', 'Long Video 1', 'Frame Video 2', 'Lat Video 2', 'Long Video 2'])
for i in range(len(row_ind)):
    resultados = pd.DataFrame({
        'Frame Video 1': video1.iloc[row_ind]['frame'].values,
        'Lat Video 1': video1.iloc[row_ind]['lat'].values,
        'Long Video 1': video1.iloc[row_ind]['long'].values,
        'Frame Video 2': video2.iloc[col_ind]['frame'].values,
        'Lat Video 2': video2.iloc[col_ind]['lat'].values,
        'Long Video 2': video2.iloc[col_ind]['long'].values,
        'Distance': distances[row_ind[i], col_ind[i]],
    })

resultados.to_csv('frames_relation.csv', index=False)
# tarda20 min'''




import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment


# Función para calcular la distancia Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# Distancia Euclidiana
def euclidean_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)


# Suponemos que cada frame representa un paso de tiempo constante.
# Si tienes marcas de tiempo reales para cada frame, deberías utilizar esas en su lugar.
video1 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/GX010042-frame_gps_interp.csv', header=0)
#video2 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/video1/video1-frame_gps_interp.csv', header=0)
video2 = pd.read_csv('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010052/GX010052-frame_gps_interp.csv', header=0)

# Preparar matrices de distancias
distances_haversine = np.zeros((len(video1), len(video2)))
distances_euclidean = np.zeros((len(video1), len(video2)))

# Calcular las distancias
for i in range(len(video1)):
    for j in range(len(video2)):
        lat1, lon1 = video1.iloc[i]['lat'], video1.iloc[i]['long']
        lat2, lon2 = video2.iloc[j]['lat'], video2.iloc[j]['long']

        distances_haversine[i, j] = haversine(lat1, lon1, lat2, lon2)
        distances_euclidean[i, j] = euclidean_distance(lat1, lon1, lat2, lon2)
        print('i:', i, 'j:', j)

# Función para guardar los resultados
def save_matched_frames(distances, filename):
    row_ind, col_ind = linear_sum_assignment(distances)

    matched_frames = pd.DataFrame({
        'Frame Video 1': video1.iloc[row_ind].reset_index(drop=True)['frame'],
        'Frame Video 2': video2.iloc[col_ind].reset_index(drop=True)['frame'],
        'Lat Video 1': video1.iloc[row_ind].reset_index(drop=True)['lat'],
        'Long Video 1': video1.iloc[row_ind].reset_index(drop=True)['long'],
        'Lat Video 2': video2.iloc[col_ind].reset_index(drop=True)['lat'],
        'Long Video 2': video2.iloc[col_ind].reset_index(drop=True)['long'],
        'Distance': [distances[row, col] for row, col in zip(row_ind, col_ind)]
    })

    matched_frames.to_csv(filename, index=False)

# Guardar los resultados
save_matched_frames(distances_haversine, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/matched_frames/matched_frames_haversine_1_3.csv')
save_matched_frames(distances_euclidean, '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/matched_frames/matched_frames_euclidean_1_3.csv')

print("Matching completado y guardado en archivos CSV")