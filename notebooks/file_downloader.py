# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import requests
import json
import os

IMAGES_FOLDER = '../arval/downloaded_images'
JSON_FOLDER = '../arval/json_files'
# Créez un dossier pour stocker les images et les fichiers json
if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)
if not os.path.exists(JSON_FOLDER):
    os.makedirs(JSON_FOLDER)

# Lire le fichier CSV
df = pd.read_csv('../arval/links_to_dataset/Result_24.csv')  # Remplacez 'votre_fichier.csv' par le nom de votre fichier

# Parcourir chaque ligne du DataFrame
for index, row in df.iterrows():
    id = row['id']
    plateNumber = row['plateNumber']
    plateNumberId = row['id_2']
    filename = row['filename']
    url = row['url']

    # Télécharger l'image
    img_data = requests.get(url).content
    with open(f'{IMAGES_FOLDER}/{plateNumber}_{filename}', 'wb') as img_file:
        img_file.write(img_data)
    
    # Créer le fichier JSON
    data = {
        'id': id,
        'plateNumber': plateNumber,
        'plateNumberId': plateNumberId,
        'filename': filename,
        'url': url
    }
    with open(f'{JSON_FOLDER}/{plateNumber}_{filename.split(".")[0]}.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)

# -

filename

# !ls ../arval

# !pip install pandas


