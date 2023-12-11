import json
import numpy as np
import os
import csv
from pycocotools import mask as maskUtils
def create_single_channel_image(data, class_names, image_size):
    # Create a dictionary to store the masks for different classes
    masks = {class_name: np.zeros((image_size[0], image_size[1]), dtype=np.uint8) for class_name in class_names}

    for i, ann in enumerate(data['annotations']):
        bitmask = maskUtils.decode(ann['segmentation'])
        class_name = ann['class_name']

        if class_name in class_names:
            masks[class_name][bitmask > 0] = i + 1

    return masks

def get_density(directory_path, output_path):
    # Directory path containing JSON files
    #directory_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/output_vegetation_small/'

    # Create a set to store all unique class names from all files
    all_class_names = set()

    # Create a dictionary to store the results for each image
    all_image_results = {}

    json_files = sorted([file for file in os.listdir(directory_path) if file.endswith('.json')])

    # Loop through all files in the directory
    for file in json_files:
        if file.endswith('.json'):
            json_path = os.path.join(directory_path, file)

            # Extract the image name without the '.json' extension
            image_name = os.path.splitext(os.path.basename(json_path))[0]

            # Read the content of the JSON file
            with open(json_path, 'r') as f:
                contenido_json = f.read()

            # Load the JSON data
            data = json.loads(contenido_json)

            # Obtener las anotaciones de segmentación
            if 'annotations' in data and len(data['annotations']) > 0:
                segmentations = data['annotations'][0]['segmentation']
                image_size = data['annotations'][0]['segmentation']['size']  # Get the image size here

                # Extract the class names from the data and add them to all_class_names set
                class_names = set(ann['class_name'] for ann in data['annotations'])
                all_class_names.update(class_names)

                # Create single channel images for each class
                masks = create_single_channel_image(data, class_names, image_size)

                # Calculate the total number of pixels in the image
                total_pixels = image_size[0] * image_size[1]

                for class_name, single_channel_img in masks.items():
                    indices_pixeles_distintos_de_cero = np.where(single_channel_img != 0)
                    total_pixeles_distintos_de_cero = len(indices_pixeles_distintos_de_cero[0])
                    density = total_pixeles_distintos_de_cero / total_pixels

                    if image_name not in all_image_results:
                        all_image_results[image_name] = {'Total Pixels': 0}

                    all_image_results[image_name][f'{class_name} Pixels'] = total_pixeles_distintos_de_cero

                    # Update the total density for the image
                    all_image_results[image_name]['Total Pixels'] += total_pixeles_distintos_de_cero
            else:
                print(f"No se encontraron anotaciones válidas en el archivo: {json_path}")
                # Set all densities to 0 for this image since there are no annotations
                if image_name not in all_image_results:
                    all_image_results[image_name] = {'Total Pixels': 0}
                for class_name in all_class_names:
                    all_image_results[image_name][f'{class_name} Pixels'] = 0

        # Save the results to a CSV file
        #output_csv_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/densities/results.csv'
        '''with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['Image Name', 'Total Pixels'] + [f'{class_name} Pixels' for class_name in all_class_names]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for image_name in sorted(all_image_results.keys()):
                results = all_image_results[image_name]
                writer.writerow({'Image Name': image_name, 'Total Pixels': results['Total Pixels'],
                                 **{f'{class_name} Pixels': results.get(f'{class_name} Pixels', 0) for class_name in all_class_names}})
        '''

if __name__ == "__main__":
    directory_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation/'
    output_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/Density/results/video1_pixels.csv'
    get_density(directory_path, output_path)
