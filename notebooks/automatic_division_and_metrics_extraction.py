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
import matplotlib.pyplot as plt
import random
import cv2
import os
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import PIL
from PIL import ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance as l_distance
import math

import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')


# +
#OCR Function

def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]

def get_words_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    text_coordinates_and_word = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:                
                converted_coordinates = convert_coordinates(obj3["geometry"],page_dim)
                
                #print("{}: {}".format(converted_coordinates,obj3["value"]))
                text_coordinates.append(converted_coordinates)
                text_coordinates_and_word.append((converted_coordinates,obj3["value"]))
    return text_coordinates,text_coordinates_and_word


def convert_to_cartesian(bound, img_dims):
    '''Convert data from matrix-like coordinates system (x,y), with y from top to bottom to a 
    cartesian coordinates system (x,y) with y from bottom to top.
    Return coordinates with the format x_min, x_max, y_min, y_max'''
    _, img_height = img_dims
    cartesian_bound = []

    for box, word in bound:
        # Convert y-values by subtracting them from the image height
        x_min, x_max, y_min, y_max = box[0], box[1], img_height - box[3], img_height - box[2]
        new_box = [x_min, x_max, y_min, y_max]
        cartesian_bound.append((new_box, word))

    return cartesian_bound

def find_next_right_word(bounding_boxes, key_word, distance_margin=1,verbose=False):
    
    key_box, key_word = _get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin,verbose)
    
    # If the word isn't found, return None
    if key_box is None:
        print("No box corresponding to the key word found")
        return None
    
    # Variables to store the closest word and its distance
    closest_word = None
    closest_box = None
    closest_distance = float('inf')
    key_x_min, key_x_max, key_y_min, key_y_max = key_box
    for box, word in bounding_boxes:
        # Check if the word's y-coordinates overlap with the key word's
        (b_x_min, b_x_max, b_y_min, b_y_max) = box
        if _has_overlap(key_y_min, key_y_max, b_y_min, b_y_max):
            # Check if the word is to the right of the key word
            if b_x_min > key_x_min: #There maybe x overlap as well
                distance = b_x_max - key_x_max  # Calculate distance right coordinates
                print(f'distance to {word}',distance)
                if distance < closest_distance:
                    closest_distance = distance
                    closest_word = word
                    closest_box = box

    return {'detected': (key_box, key_word),
            'next':(closest_box, closest_word)}

def _compare_words_similarity(word1, word2, distance_margin):
    return l_distance(word1, word2) <= distance_margin

def _has_overlap(y1_min, y1_max, y2_min, y2_max, minimum_overlap=5):
    return (y2_min  <= y1_min <= y2_max - minimum_overlap) or \
           (y2_min + minimum_overlap <= y1_max <= y2_max)

def _get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin, verbose):
    # Find the bounding box for the search word
    key_box = None
    #print(bounding_boxes)
    for box, word in bounding_boxes:
        if _compare_words_similarity(word.lower(), key_word.lower(), distance_margin):
            #print(word, key_word)
            key_box = box
            break
    if verbose:
        print(f'Key box defined to {key_box}')
    
    return key_box, key_word


def merge_word_with_following(converted_boxes, key_word):
    """
    Modified converted_boxes to remove the key_word/key_box tuple and the next_word/next_box tuple 
    and replace it by a tuple with concatenated key words and merged boxes (taking the maximum extent). 
    """
    data = find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=True)
    new_word = data['detected'][1] + ' ' + data['next'][1]
    new_box = [min(data['detected'][0][0],data['next'][0][0]),
               max(data['detected'][0][1],data['next'][0][1]),
               min(data['detected'][0][2],data['next'][0][2]),
               max(data['detected'][0][3],data['next'][0][3])]

    index_to_remove1 = next((i for i, v in enumerate(converted_boxes) if v[0] == data['detected'][0]), None)
    index_to_remove2 = next((i for i, v in enumerate(converted_boxes) if v[0] == data['next'][0]), None)

    # If the item is found, remove it using pop
    if index_to_remove1 is not None:
        converted_boxes.pop(index_to_remove1)
    if index_to_remove2 is not None:
        converted_boxes.pop(index_to_remove2)
    
    converted_boxes.append((new_box,new_word))
    return converted_boxes

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
    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    if find_next_right_word(converted_boxes,'Restitué') is not None:
        converted_boxes = merge_word_with_following(converted_boxes,'Restitué')
    return converted_boxes,image_dims

def postprocess_boxes_and_words(converted_boxes, block, safe, verbose):
    converted_boxes = remove_word(converted_boxes, ":")
    if block == 'block_2':
        converted_boxes = merge_word_with_following(converted_boxes,'Restitué', safe, verbose)
        converted_boxes = merge_word_with_following(converted_boxes,'Lieu', safe, verbose)
    if block == 'block_5':
        converted_boxes = merge_word_with_following(converted_boxes,'Nom', safe, verbose)

    return converted_boxes


# +
#Function find line and cut the doc:

import random

from bisect import bisect_left

def find_closest_lines(all_lines, reference_lines):
    closest_lines = []

    for ref_line in reference_lines:
        # Find the insertion point where the line would go in the sorted list
        insert_point = bisect_left(all_lines, ref_line)
        # Find the closest line by comparing the insertion point and the previous line
        if insert_point == 0:
            closest_lines.append(all_lines[0])
        elif insert_point == len(all_lines):
            closest_lines.append(all_lines[-1])
        else:
            before = all_lines[insert_point - 1]
            after = all_lines[insert_point]
            closest_lines.append(before if (ref_line - before) <= (after - ref_line) else after)

    return closest_lines



def find_max_spacing_non_crossing_lines(converted_box, img_height, line_number=4):
    # Parse the bounding boxes into a simpler format
    bounds = [(box[0][2], box[0][3]) for box in converted_box]

    # Define a function to check if a line crosses a bounding box
    def does_line_cross_box(y, box):
        y1, y2 = box
        return y1 <= y <= y2

    # Iterate through possible y-coordinates (lines)
    non_crossing_lines = [y for y in range(img_height) if not any(does_line_cross_box(y, b) for b in bounds)]

    # Sort the non-crossing lines based on their position
    non_crossing_lines.sort()

    theoretical_dividing_number = 0
    theoretical_dividing = []
    for i in range(line_number):
        theoretical_dividing_number += (img_height/line_number)
        #print(theoretical_dividing_number)
        theoretical_dividing.append(theoretical_dividing_number)

    selected_lines = find_closest_lines(non_crossing_lines, theoretical_dividing)
    
    return selected_lines

def cut_and_save_image(input_image_path, output_folder, selected_lines):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the input image
    image = cv2.imread(input_image_path)

    if image is None or image.size == 0:
        print("Error: Failed to read the input image.")
        return

    # Add the top of the image as the starting line
    selected_lines.insert(0, 0)

    # Loop through the selected lines and cut the image accordingly
    for i, y1 in enumerate(selected_lines):  
        if i < len(selected_lines) - 1:
            y2 = selected_lines[i+1]
        else:
            return

        # Crop the image
        print('y1 =',y1,'y2 =',y2)
        cropped_image = image[y1:y2, :]

        # Generate an output filename
        output_filename = os.path.join(output_folder, f"output_{i + 1}.jpg")

        # Check if cropped_image is empty
        if cropped_image is not None and cropped_image.size != 0:
            # Save the cropped image
            cv2.imwrite(output_filename, cropped_image)
            print(f"Saved {output_filename}")
        else:
            print(f"Error: Cropped image is empty for Line {i + 1}")

# +
#Print function:


def plot_boxes(bound, img_dims):
    # Extract the dimensions of the image
    img_width, img_height = img_dims
    
    for b in bound:
        b1 = b[0]
        # Extract the coordinates of the bounding box
        x1, x2, y1, y2 = b1
        # Define the four corners of the box
        box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        # Extract x and y coordinates separately
        x_coords, y_coords = zip(*box_corners)
        # Plot the bounding box
        plt.plot(x_coords, y_coords, color='red')
        
    plt.xlim(0, img_width)
    plt.ylim(0, img_height)  # Start the y-axis at the top to match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bounding Boxes')
    plt.show()

def plot_boxes_and_lines(bound, img_dims, non_crossing_lines=None):
    # Extract the dimensions of the image
    img_width, img_height = img_dims
    
    # Create a random color for the non-crossing lines
    non_crossing_line_color = "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    for b in bound:
        b1 = b[0]
        # Extract the coordinates of the bounding box
        x1, x2, y1, y2 = b1
        # Define the four corners of the box
        box_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        # Extract x and y coordinates separately
        x_coords, y_coords = zip(*box_corners)
        # Plot the bounding box in red
        plt.plot(x_coords, y_coords, color='red')

    # Plot the non-crossing horizontal lines in a random color
    if non_crossing_lines:
        for y in non_crossing_lines:
            plt.axhline(y, color=non_crossing_line_color, linestyle='--', label="Non-Crossing Lines")

    plt.xlim(0, img_width)
    plt.ylim(0, img_height)  # Start the y-axis at the top to match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bounding Boxes and Non-Crossing Lines')
    plt.legend()
    plt.show()


# -

# ## I. Running the detection & recognition model

img_path = "data/performances_data/arval_classic_restitution_images/DH-427-VH_PV_RESTITUTION_DH-427-VH_p1.jpeg"
#img_path = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/DM-984-VT_Proces verbal de restitution_page-0001.jpg"
converted_boxes,image_dims = get_processed_boxes_and_words(img_path)
#print(img_path)


plot_boxes(converted_boxes, image_dims)
print(image_dims)
image_dims[1]

# ## II. Dividing image in bloc 

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

def subdivising_image(input_img,out_bloc_folder,line_number):
    converted_boxes,image_dims = get_processed_boxes_and_words(img_path)
    img_height = image_dims[1]
    non_crossing_lines = find_max_spacing_non_crossing_lines(converted_boxes, img_height,line_number)
    sorted_lines = sorted(non_crossing_lines)
    cut_and_save_image(img_path, out_bloc_folder, sorted_lines)


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
        
    path_to_name_jpeg = local_fold/jpg_files[0]
    print(path_to_name_jpeg)

    subfolder_name = "automaticbloc"

    # Create the subfolder if it doesn't exist
    subfolder_path = os.path.join(local_fold, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.mkdir(subfolder_path)
    subdivising_image(path_to_name_jpeg,subfolder_path,4)

    
# -

# ## III. Performance estimation for one doc

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

