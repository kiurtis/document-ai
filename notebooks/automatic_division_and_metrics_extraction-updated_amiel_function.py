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

# # Main functions

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
import math
from pprint import pprint 
import PIL
from PIL import ImageDraw

from Levenshtein import distance as l_distance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from performance_estimation import compute_metrics_for_multiple_jsons, get_result_template, perform_hyperparameter_optimization, parse_random_search_results
from document_parsing import get_words_coordinates, convert_to_cartesian, merge_word_with_following, find_next_right_word
from plotting import plot_boxes_with_text, print_blocks_on_document, plot_centroids,plot_boxes,plot_boxes_and_lines
from pre_ocr_division import find_max_spacing_non_crossing_lines,cut_and_save_image

# +

img_path = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/blocks/DM-984-VT_Proces_verbal_de_restitution_page-0001_block 2.png"

img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()

image_dims = (output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output,verbose=False)

converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
converted_boxes = merge_word_with_following(converted_boxes,'Restitué')


# +
#My function, see what I do with it 
def run_doctr_model(img_path):
    ''' Load an image and run the model on it.'''
    img = DocumentFile.from_images(img_path)
    model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)
    result = model(img)
    output = result.export()
    return output
    
def get_processed_boxes_and_words(img_path):
    ''' Load an image, run the model, get the box words pairs and preprocess the needed words.'''
    output = run_doctr_model(img_path)
    image_dims = (output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])
    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output,verbose=False)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    if find_next_right_word(converted_boxes,'Restitué') is not None:
        converted_boxes = merge_word_with_following(converted_boxes,'Restitué')
    return converted_boxes,image_dims


# -

# ## I. Automatic division single image

img_path = "data/performances_data/arval_classic_restitution_images/DH-427-VH_PV_RESTITUTION_DH-427-VH_p1.jpeg"
#img_path = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/DM-984-VT_Proces verbal de restitution_page-0001.jpg"
converted_boxes,image_dims = get_processed_boxes_and_words(img_path)


plot_boxes(converted_boxes, image_dims)

img_height = image_dims[1]
line_num = 4
non_crossing_lines= find_max_spacing_non_crossing_lines(converted_boxes, img_height,line_num)
print("Non-crossing lines:", non_crossing_lines)

# +
plot_boxes_and_lines(converted_boxes, image_dims,non_crossing_lines)
output_fold = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/autotest"
output_fold= "data/performances_data/arval_classic_restitution_images/autotest"

cut_and_save_image(img_path, output_fold, non_crossing_lines)


# -

# ## II. Automatic division batch of image

def subdivising_image(input_img,out_bloc_folder,line_number):
    converted_boxes,image_dims = get_processed_boxes_and_words(input_img)
    img_height = image_dims[1]
    non_crossing_lines = find_max_spacing_non_crossing_lines(converted_boxes, img_height,line_number)
    sorted_lines = sorted(non_crossing_lines)
    cut_and_save_image(input_img, out_bloc_folder, sorted_lines)


# +
from pathlib import Path
FOLDER_IMAGES = Path('data/performances_data/fleet_services_images')

def clean_listdir(path_to_folder):
    return [listed for listed in os.listdir(path_to_folder) if not listed.startswith('.')]

image_list = clean_listdir(FOLDER_IMAGES)
#print(image_list)

#image_list= image_list[0:2]

for element in image_list:
    print(f'==== Running for file: {element} =====')
    filename_prefix = f'{element[:-5]}'

    local_fold = FOLDER_IMAGES/element
    files_in_folder = os.listdir(local_fold)
    
    # Filter files with ".jpg" extension
    jpg_files = [file for file in files_in_folder if (file.endswith(".jpg") or file.endswith(".jpeg"))]
    if len(jpg_files) > 1:
        raise ValueError("More than one JPG file found in the folder.")
    elif len(jpg_files) == 0:
        raise ValueError("No JPG files found in the folder.")

    # If there's only one JPG file, print its name
    if len(jpg_files) == 1:
        print("Found the JPG file:", jpg_files[0])
        
    path_to_name_jpeg = str(local_fold/jpg_files[0])
    print(path_to_name_jpeg)

    subfolder_name = "automaticbloc"

    # Create the subfolder if it doesn't exist
    subfolder_path = os.path.join(local_fold, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
    subdivising_image(path_to_name_jpeg,subfolder_path,4)

    
# -

# ## III. Performance estimation for one doc (To do)

# +
import json
from pathlib import Path
FOLDER_GROUND_TRUTHS = Path('data/performances_data/fleet_services_jsons')
FOLDER_IMAGES = Path('data/performances_data/fleet_services_images')
VERBOSE = True
def clean_listdir(path_to_folder):
    return [listed for listed in os.listdir(path_to_folder) if not listed.startswith('.')]

def read_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data
    
def get_result_template():
    json_template = {}
    sample_json = read_json(FOLDER_GROUND_TRUTHS/f'{ground_truths_list[0]}')
    block_names = [k for k in sample_json.keys() if k.startswith('block')]
    for bn in block_names:
        json_template[bn] = sample_json[bn].keys()
    return json_template


def has_found_box(value):
    return type(value) == tuple


# -

# List of data
image_list = clean_listdir(FOLDER_IMAGES)
ground_truths_list = [x + '.json' for x in image_list]
print(ground_truths_list)
print(image_list)

# +
result_template = get_result_template()
#print(result_template)

def flatten_dict(d, sep='_'):
    flattened = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, sep=sep))
        else:
            flattened[k] = v
    return flattened
    
actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]

for dico in actual_json_list:
    print('.    ')
    dico_flat = flatten_dict(dico)
    print(dico_flat)
    print('.    ')

def get_flatten_result_template():
    actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]
    sample_json = flatten_dict(actual_json_list[0])
    json_template = sample_json.keys()
    return json_template

result_template_flat = get_flatten_result_template()

# Convert it to a list
result_template_flat_list = list(result_template_flat)

# Remove 'File Name' from the list
if 'File Name' in result_template_flat_list:
    result_template_flat_list.remove('File Name')

print(result_template)


# +
all_results = []
#image_list= image_list[1:2]
print(image_list)

for element in image_list:
    print(f'==== Running for file: {element} =====')
    filename_prefix = f'{element[:-5]}'

    result_json = {}
    result_json['File Name'] = element
    
    folder_with_blocs = FOLDER_IMAGES/element/'automaticbloc'
    bloc_list = clean_listdir(folder_with_blocs)
    print(bloc_list)

    for bloc in bloc_list:
        #print(bloc)
        path_to_name_jpeg = FOLDER_IMAGES/element/'automaticbloc'/bloc
        print(path_to_name_jpeg)

        converted_boxes,image_dims = get_processed_boxes_and_words(img_path=path_to_name_jpeg)
        #print(converted_boxes)
        
        #converted_boxes = postprocess_boxes_and_words(converted_boxes)
        
        for key_word in result_template_flat_list:
            print(f'Running {key_word}')
            result_json[key_word] = find_next_right_word(converted_boxes, key_word)
            print(result_json[key_word])
            if has_found_box(result_json[key_word]):
                result_json[key_word] = result_json[key_word]['next']
    all_results.append(result_json)
    #print(all_results)
    



# +
all_results = []
#image_list= image_list[1:2]
print(image_list)

for element in image_list:
    print(f'==== Running for file: {element} =====')
    filename_prefix = f'{element[:-5]}'

    result_json = {}
    result_json['File Name'] = element
    
    folder_with_blocs = FOLDER_IMAGES/element/'automaticbloc'
    bloc_list = clean_listdir(folder_with_blocs)
    print(bloc_list)

    for bloc in bloc_list:
        #print(bloc)
        path_to_name_jpeg = FOLDER_IMAGES/element/'automaticbloc'/bloc
        print(path_to_name_jpeg)

        converted_boxes,image_dims = get_processed_boxes_and_words(img_path=path_to_name_jpeg)
        #print(converted_boxes)

        for bn in list(result_template.keys()):
            result_json[bn] = {}

            for key_word in result_template[bn]:
                print(f'Running {key_word}')
                result_json[bn][key_word] = find_next_right_word(converted_boxes, key_word, 
                                                             distance_margin=2, verbose=VERBOSE)
                if has_found_box(result_json[bn][key_word]):
                    result_json[bn][key_word] = result_json[bn][key_word]['next']
    all_results.append(result_json)
    


        
        
        #converted_boxes = postprocess_boxes_and_words(converted_boxes)
        
        #for key_word in result_template_flat_list:
        #    print(f'Running {key_word}')
        #    result_json[key_word] = find_next_right_word(converted_boxes, key_word)
        #    print(result_json[key_word])
        #    if has_found_box(result_json[key_word]):
          #      result_json[key_word] = result_json[key_word]['next']
    #all_results.append(result_json)
    #print(all_results)
    
print(all_results)


# +
def clean_predicted_data(data):
    new_data = {}
    new_data['File Name'] = data['File Name']

    for block, values in [(el,_) for (el,_) in data.items() if el.startswith('block')]:
        new_data[block] = {}
        if isinstance(values, dict):
            for key, content in values.items():
                if isinstance(content, dict) and 'next' in content:
                    new_data[block][key] = content['next'][1]
                else:
                    new_data[block][key] = content
    return new_data

predicted_dict_list = [clean_predicted_data(results) for results in all_results]
print(predicted_dict_list)
# -

actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]

# +
from performance_estimation import compute_metrics_for_multiple_jsons


metrics = compute_metrics_for_multiple_jsons(predicted_dict_list, actual_json_list)
print(metrics)

# -

# ## IV Perf estimation for the all batch

actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]

# +
from performance_estimation import compute_metrics_for_multiple_jsons


metrics = compute_metrics_for_multiple_jsons(predicted_dict_list, actual_json_list)
print(metrics)

