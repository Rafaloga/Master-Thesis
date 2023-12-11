import pandas as pd
import matplotlib.pyplot as plt

# Lee el archivo CSV en un DataFrame de Pandas
df = pd.read_csv("/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010042/GX010042-frame_gps_interp.csv")

# Crea una lista de coordenadas latitud y longitud
latitudes = df['lat']
longitudes = df['long']


# Crea un bucle para crear múltiples mapas
for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
    # Crea un nuevo gráfico
    plt.figure(figsize=(10, 10))

    # Dibuja todas las coordenadas en el mapa
    plt.scatter(longitudes, latitudes, marker='o', c='green')

    # Dibuja un punto más fuerte para la coordenada actual
    plt.scatter(lon, lat, c='red', s=100)

    # Añade etiquetas y título
    #plt.xlabel('Longitud')
    #plt.ylabel('Latitud')
    #plt.title(f'Mapa {i}')
    plt.legend()

    # Guarda el mapa en un archivo o muéstralo en pantalla
    plt.savefig(f'maps/GX010042/mapa_{i}.png')
    # plt.show()  # Descomenta esta línea para mostrar el mapa en pantalla

    # Cierra el gráfico actual
    plt.close()

    if i == 100:
        break

print("Mapas generados exitosamente.")
