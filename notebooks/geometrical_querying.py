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

# # First Test

# +
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
# -

# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from performance_estimation import perform_hyperparameter_optimization, parse_random_search_results
from ai_documents.analysis.cv.document_parsing import get_words_coordinates, convert_to_cartesian, merge_word_with_following, find_next_right_word
from ai_documents.plotting import print_blocks_on_document, plot_centroids

# -

# ## I. Running the detection & recognition model

# +
#img_path = "../arval/barth_jpg/arval_fleet_service_restitution/DY-984-XY_PV de reprise_p2.jpeg"

img_path = "data/test_data/DM-984-VT_Proces verbal de restitution_page-0001_bloc_2.png"

img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()
# -


print_blocks_on_document(output, img_path)

# ## II. Preprocessing 

# Getting all the words and converting their bounding box to cartesian coords (y from the bottom to the top).

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output,verbose=True)
image_dims = (
    output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])
converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
print(converted_boxes)


# Preprocessing of the key words composed of multiple words (for instance "Restitué" -> "Restitué le").

# +
img_path = "data/test_data/DM-984-VT_Proces verbal de restitution_page-0001_bloc_2.png"

img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output,verbose=False)


converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
merge_word_with_following(converted_boxes,'Restitué')
# -

# ## III. Getting the value for a given key

key_word = ""
print(find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=True))

plot_centroids(converted_boxes, image_dims)

# ## IV Hyperparameter optimization

# +
# Define hyperparameter space
FOLDER_GROUND_TRUTHS = Path('data/performances_data/fleet_services_jsons')
FOLDER_IMAGES = Path('data/performances_data/fleet_services_images')
VERBOSE = True
# List of data
#image_list = clean_listdir(FOLDER_IMAGES, only="dir")
#ground_truths_list = [x + '.json' for x in image_list]

# Define number of iterations and results file path
NUM_ITERATIONS = 20
RESULTS_LIST_PATH = 'hyperparameters_optimization_results.json'

DET_ARCHS = [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
    ]
DET_ROT_ARCHS = ["db_resnet50_rotation"]

RECO_ARCHS = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "sar_resnet31",
    "master",
    "vitstr_small",
    "vitstr_base",
    "parseq",
]

HYPERPARAMETER_SPACE = {'det_arch':DET_ARCHS + DET_ROT_ARCHS,
        'reco_arch':RECO_ARCHS,
        'pretrained':[True, ],
        'distance_margin': [1, 2, 5, 10, 20], # find_next_right_word for words_similarity
        'max_distance': [10, 50, 100, 200, 500], # find_next_right_word
        'minimum_overlap': [1, 2, 5, 10, 20, 50, 100] # find_next_right_word for _has_overlap
}

hyperparameters = {k:v[0] for (k,v) in HYPERPARAMETER_SPACE.items()}

perform_hyperparameter_optimization(num_iterations=NUM_ITERATIONS, 
                                    hyperparameter_space=HYPERPARAMETER_SPACE, 
                                    folder_images=FOLDER_IMAGES, 
                                    results_list_path=RESULTS_LIST_PATH,
                                    verbose=VERBOSE)

# -

# ## V Performance estimation

# +
# Usage example:
results_file_path = 'hyperparameters_optimization_results.json'
best_hyperparameters_by_section = parse_random_search_results(results_file_path)

# Print the best hyperparameters for each section
for section, data in best_hyperparameters_by_section.items():
    print(f"Section: {section}")
    if section == 'general':
        for metric, result in data.items():
            print(f"  Best {metric}: {result['value']}")
            print(f"  Hyperparameters: {result['hyperparameters']}\n")
    else:
        for identifier, metrics in data.items():
            print(f"  {identifier}:")
            for metric, result in metrics.items():
                print(f"    Best {metric}: {result['value']}")
                print(f"    Hyperparameters: {result['hyperparameters']}\n")

# -

best_hyperparameters_by_metric

# +
#TODO: Identify a way to distinguish the "le" from "le conducteur" and for the date.
