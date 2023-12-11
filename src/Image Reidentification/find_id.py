import csv
import pandas as pd
import json
import os
import json
from collections import OrderedDict
# Lee los datos del primer archivo CSV
lista1 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_GX010042.csv'
lista2 = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Homography/resultados/crops_list_orden/resultados_crops_video1.csv'
# Función para leer los datos de un archivo CSV y devolverlos como lista de diccionarios
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

# Función para procesar las relaciones
def process_relations(data):
    relations = []
    id_counter = 0
    ids = {}  # Mapeo de segmentos a ID

    # Asignar IDs iniciales
    for segment in data[0]['mixed_list']:
        ids[segment] = id_counter
        id_counter += 1

    # Procesar las filas de datos
    for frame_number, row in enumerate(data):
        next_ids = {}
        frame = row['img'].split('.')[0]  # Obtenemos el número de frame sin la extensión .jpg

        # Relacionar con la siguiente fila
        for actual, next_ in zip(row['actual_frame_list'], row['next_frame_list']):
            if actual in ids:
                relations.append({
                    'id': ids[actual],
                    'frame': frame,
                    'segment': actual,
                    'list': 'mixed_list'
                })
                next_ids[next_] = ids[actual]  # El siguiente frame tendrá el mismo ID

        # Verificar elementos nuevos que no estaban en la fila anterior
        for segment in row['mixed_list']:
            if segment not in next_ids.values():
                # Este segmento es nuevo, le asignamos un nuevo ID
                next_ids[segment] = id_counter
                id_counter += 1

        # Prepararse para la siguiente iteración
        ids = next_ids

    return relations

# Función para escribir las relaciones en un archivo CSV
def write_output_grouped_by_id(relations, output_file):
    grouped_relations = {}
    # Agrupamos las relaciones por ID
    for relation in relations:
        if relation['id'] not in grouped_relations:
            grouped_relations[relation['id']] = []
        grouped_relations[relation['id']].append(relation)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['id', 'frame_segment_list']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Escribimos las relaciones agrupadas por ID
        for id_, segments in grouped_relations.items():
            # Ordenamos los segmentos por frame antes de escribirlos
            segments_sorted = sorted(segments, key=lambda x: int(x['frame']))
            frame_segment_list = '; '.join([f"{s['frame']}:{s['segment']}" for s in segments_sorted])
            writer.writerow({'id': id_, 'frame_segment_list': frame_segment_list})


# Primero, leemos el archivo CSV y almacenamos la información GPS en un diccionario.
def read_gps_data(csv_file):
    gps_data = {}
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = row['frame']
            #print(frame)
            # Si los valores de latitud y longitud están vacíos, asigna None o un valor por defecto.
            gps_data[frame] = {
                'lat': row['lat'] if row['lat'] else None,
                'long': row['long'] if row['long'] else None,
                'azi': row['azi'] if row['azi'] else None
            }
    return gps_data

# Modificamos la función que escribe el JSON para que incluya la información GPS.
'''def write_output_to_json(relations, gps_data, output_file):
    grouped_relations = {}
    for relation in relations:
        relation_id = relation['id']
        # Asumimos que el frame en la relación viene en formato '000000.jpg' y necesitamos extraer '0'
        frame_number = relation['frame'].rstrip('.jpg').lstrip('0') or '0'  # Esto convierte '000000.jpg' en '0'
        segment = relation['segment']
        list_origin = relation.get('list_origin', 'Unknown')
        # Obtenemos las coordenadas GPS para este frame usando el número del frame.
        gps_info = gps_data.get(frame_number, {})

        grouped_relations.setdefault(relation_id, [])
        grouped_relations[relation_id].append({
            'frame': frame_number,  # Usamos el número del frame aquí
            'segment': segment,
            'list_origin': list_origin,
            'gps': gps_info  # Agregamos la información GPS
        })
    
    with open(output_file, 'w') as jsonfile:
        json.dump(grouped_relations, jsonfile, indent=4)
'''


def write_output_to_json(relations, gps_data, output_file):
    # Ordenar las relaciones por 'id'
    relations.sort(key=lambda x: x['id'])

    grouped_relations = OrderedDict()
    for relation in relations:
        relation_id = relation['id']
        frame_number = relation['frame'].rstrip('.jpg').lstrip('0') or '0'
        segment = relation['segment']
        list_origin = relation.get('list_origin', 'Unknown')
        gps_info = gps_data.get(frame_number, {})

        grouped_relations.setdefault(relation_id, [])
        grouped_relations[relation_id].append({
            'frame': frame_number,
            'segment': segment,
            'list_origin': list_origin,
            'gps': gps_info
        })

    with open(output_file, 'w') as jsonfile:
        json.dump(grouped_relations, jsonfile, indent=4)

def write_output_by_frame_to_json(relations, gps_data, output_file):
    frame_relations = {}
    for relation in relations:
        # Asumimos que el frame en la relación viene en formato '000000.jpg' y necesitamos extraer '0'
        frame_number = relation['frame'].rstrip('.jpg').lstrip('0') or '0'  # Esto convierte '000000.jpg' en '0'

        if frame_number not in frame_relations:
            frame_relations[frame_number] = {
                'frame': frame_number,  # Incluimos el número de frame en la información de salida
                'segments': [],
                'gps': gps_data.get(frame_number, {})  # Agregamos la información GPS
            }

        frame_relations[frame_number]['segments'].append({
            'id': relation['id'],
            'segment': relation['segment'],
            'list_origin': relation.get('list', 'Unknown')
        })

    # Ordenamos los frames por número antes de escribirlos en el JSON
    ordered_frames = sorted(frame_relations.keys(), key=int)
    with open(output_file, 'w') as jsonfile:
        # Crear una lista para los frames ordenados
        frames_list = [frame_relations[frame] for frame in ordered_frames]
        json.dump(frames_list, jsonfile, indent=4)


# Leer datos de los archivos CSV
data = read_csv_data(lista1)
gps_data = read_gps_data('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/GX010042-frame_gps_interp.csv')

# Procesar relaciones y obtener lista de diccionarios
#relations = process_relations(data[0:10])
relations = process_relations(data)
output_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Resultados/ID/'
json1 = 'output_grouped_GX010042_aux.json'
output_json1 = os.path.join(output_path, json1)
json2 = 'output_by_frame_GX010042.json'
output_json2 = os.path.join(output_path, json2)
# Escribir las relaciones en un archivo CSV de salida
#write_output_grouped_by_id(relations, 'output_grouped.csv')
write_output_to_json(relations, gps_data, output_json1)
# Usar la función modificada para escribir la salida
write_output_by_frame_to_json(relations, gps_data, output_json2)
