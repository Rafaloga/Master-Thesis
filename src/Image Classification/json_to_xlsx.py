import os
import json
import openpyxl

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

# Crea un nuevo archivo de Excel
workbook = openpyxl.Workbook()
worksheet = workbook.active
worksheet.title = "Datos"

# Agrega el encabezado
header = ["Archivo", "Imagen", "Score", "Nombre Científico", "Familia", "Nombres Comunes", "ID GBIF", "ID POWO", "ID IUCN", "Categoría IUCN"]
worksheet.append(header)

# Itera sobre los datos JSON y escribe cada fila en el archivo Excel
for nombre_archivo, datos in datos_totales:
    for imagen, info in datos.items():
        for clase in info["top_classes"]:
            row = [
                nombre_archivo,
                imagen,
                clase["score"],
                clase["species"]["scientificNameWithoutAuthor"],
                clase["species"]["family"]["scientificName"],
                ", ".join(clase["species"]["commonNames"]),
                clase["gbif"].get("id", ""),
                clase.get("powo", {}).get("id", ""),
                clase.get("iucn", {}).get("id", ""),
                clase.get("iucn", {}).get("category", "")
            ]
            worksheet.append(row)

# Guarda el archivo de Excel
workbook.save('vegetation_classes.xlsx')
