import requests
import json

url = 'https://my-api.plantnet.org/v2/identify/all'
headers = {
    'accept': 'application/json'
}
params = {
    'include-related-images': 'false',
    'no-reject': 'false',
    'lang': 'en',
    'type': 'kt',
    'api-key': '2b10fRljjEAFXm6GM3zm5Tmmu'
}
files = {
    'images': ('recorte_0.jpg', open('/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/crops/video1/000000/recorte_2.jpg', 'rb'), 'image/jpeg')
}
data = {
    'organs': 'auto'
}

response = requests.post(url, headers=headers, params=params, files=files, data=data)

if response.status_code == 200:
    print("Respuesta exitosa:")
    response_json = response.json()

    # Obtener las tres clases más probables
    top_classes = response_json.get('results', [])[:3]

    # Guardar las tres clases en un nuevo archivo JSON
    with open('top_classes.json', 'w') as json_file:
        json.dump(top_classes, json_file)
        print("Las tres clases más probables se han guardado en 'top_classes.json'")
else:
    print(f"Error en la solicitud: Código de estado {response.status_code}")
    print(response.text)
