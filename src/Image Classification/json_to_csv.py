import os
import json
import csv

# Directorio que contiene los archivos JSON
directorio_json = '/mnt/rhome/rlg/PROYECTO/PlantNet-API/results/video1/'

# Lista para almacenar los datos JSON de todos los archivos
datos_totales = []

# Obtén la lista de nombres de archivo en el directorio y ordénalos alfabéticamente
archivos_json = sorted(os.listdir(directorio_json))

# Recorre los archivos JSON en orden alfabético
for filename in archivos_json:
    if filename.endswith('_results.json'):
        nombre_base = os.path.splitext(filename)[0].replace('_results', '')  # Elimina "_results" y la extensión ".json"
        with open(os.path.join(directorio_json, filename), 'r') as archivo_json:
            datos_json = json.load(archivo_json)
            datos_totales.append((nombre_base, datos_json))

# Abre un archivo CSV para escribir
with open('datos.csv', 'w', newline='') as csvfile:
    fieldnames = ["Archivo", "Imagen", "Score", "Nombre Científico", "Familia", "Nombres Comunes", "ID GBIF", "ID POWO",
                  "ID IUCN", "Categoría IUCN"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Escribe el encabezado del archivo CSV
    writer.writeheader()

    # Itera sobre los datos JSON y escribe cada fila en el CSV
    for nombre_archivo, datos in datos_totales:
        for imagen, info in datos.items():
            for clase in info["top_classes"]:
                row = {
                    "Archivo": nombre_archivo,
                    "Imagen": imagen,
                    "Score": clase["score"],
                    "Nombre Científico": clase["species"]["scientificNameWithoutAuthor"],
                    "Familia": clase["species"]["family"]["scientificName"],
                    "Nombres Comunes": ", ".join(clase["species"]["commonNames"]),
                    "ID GBIF": clase["gbif"].get("id", ""),  # Usa get() para manejar claves faltantes
                    "ID POWO": clase.get("powo", {}).get("id", ""),  # Usa get() para manejar claves faltantes
                    "ID IUCN": clase.get("iucn", {}).get("id", ""),  # Usa get() para manejar claves faltantes
                    "Categoría IUCN": clase.get("iucn", {}).get("category", "")
                    # Usa get() para manejar claves faltantes
                }
                writer.writerow(row)
