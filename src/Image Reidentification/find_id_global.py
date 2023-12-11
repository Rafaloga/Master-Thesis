import csv
import pandas as pd
import json
def read_csv_data(file_name):
    data = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convertimos las cadenas a listas usando eval
            row['mixed_list'] = eval(row['mixed_list'])
            row['different_videos_list'] = eval(row['different_videos_list'])
            row['actual_frame_list'] = eval(row['actual_frame_list'])
            row['next_frame_list'] = eval(row['next_frame_list'])
            data.append(row)
    return data
# Esta función busca la correspondencia de un segmento en otro video y retorna el id correspondiente si existe
def find_corresponding_id(segment_id, global_relations, other_video_id):
    for key, value in global_relations.items():
        for segment in value['segments']:
            if segment['video_id'] == other_video_id and segment['corresponding_segment'] == segment_id:
                return key
    return None


# Esta función procesará un solo video y actualizará las relaciones globales.
def process_video(video_data, global_relations, current_video_id, other_video_id):
    for data in video_data:
        print(data['mixed_list'])
        frame = data['img'].rstrip('.jpg').lstrip('0') or '0'
        mixed_list = data['mixed_list']  # Convertimos el string a una lista

        # Procesa los datos y actualiza el diccionario global_relations.
        for segment_id in mixed_list:
            corresponding_id = find_corresponding_id(segment_id, global_relations, other_video_id)

            # Si encontramos una correspondencia, usamos el mismo ID para la segmentación en ambos videos.
            if corresponding_id is not None:
                global_relations[corresponding_id]['segments'].append({
                    'video_id': current_video_id,
                    'frame': frame,
                    'segment_id': segment_id,
                    'corresponding_segment': corresponding_id
                })
            else:
                # Si no hay correspondencia, generamos un nuevo ID y lo añadimos al global_relations.
                new_id = max(global_relations.keys()) + 1 if global_relations else 0
                global_relations[new_id] = {
                    'segments': [{
                        'video_id': current_video_id,
                        'frame': frame,
                        'segment_id': segment_id,
                        'corresponding_segment': segment_id
                    }]
                }


# Diccionario para almacenar las relaciones globales
global_relations = {}

lista1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_GX010042.csv'
lista2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list/resultados_crops_video1.csv'
# Supongamos que estos son tus datos, reemplaza esto con la carga real de tus datos

# Leer datos de los archivos CSV
data1 = read_csv_data(lista1)
data2 = read_csv_data(lista2)
video0_data = data1[0:10]
video1_data = data2[0:10]

# Procesamos el primer video, asumiendo que es el video0
process_video(video0_data, global_relations, 'video0', 'video1')
# Procesamos el segundo video, asumiendo que es el video1
process_video(video1_data, global_relations, 'video1', 'video0')

# Función para escribir los resultados en un archivo JSON
import json


def write_output_to_json(output_data, filename):
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=4)


# Llamada a la función para escribir los resultados
write_output_to_json(global_relations, 'output_combined_global.json')