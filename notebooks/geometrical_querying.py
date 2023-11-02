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

# +
import math
from pprint import pprint 

from pdf2image import convert_from_path
import PIL
from PIL import ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance as l_distance
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def convert_pdf_to_jpg(pdf_path, output_folder=None, dpi=300):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path, dpi)

    # Save images to the output directory
    for i, image in enumerate(images):
        image_name = f"output_page_{i + 1}.jpg"
        if output_folder:
            image_name = f"{output_folder}/{image_name}"
        image.save(image_name, "JPEG")


def get_block_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    i = 0
    for obj1 in output['pages'][0]["blocks"]:              
        converted_coordinates = convert_coordinates(
                                           obj1["geometry"],page_dim
                                          )
        print("{}: {}".format(converted_coordinates,
                                      i
                                      )
                     )
        text_coordinates.append(converted_coordinates)
        i+=1
    return text_coordinates

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

def get_words_coordinates(output, verbose):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    text_coordinates_and_word = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:                
                converted_coordinates = convert_coordinates(obj3["geometry"],page_dim)
                if verbose:
                    print("{}: {}".format(converted_coordinates,obj3["value"]))
                text_coordinates.append(converted_coordinates)
                text_coordinates_and_word.append((converted_coordinates,obj3["value"]))
    return text_coordinates,text_coordinates_and_word

def draw_bounds(image, bound):
    draw = ImageDraw.Draw(image)
    for b in bound:
        p0, p1, p2, p3 = [b[0],b[2]], [b[1],b[2]], \
                         [b[1],b[3]], [b[0],b[3]]
        draw.line([*p0,*p1,*p2,*p3,*p0], fill='blue', width=2)
    return image

def plot_centers2(bound, img_dims):
    # Extract the dimensions of the image
    img_width, img_height = img_dims
    
    for b in bound:
        b1 = b[0]
        c = [(b1[0] + b1[1])/2, (b1[2] + b1[3])/2]
        plt.scatter(c[0], c[1], color='red')
        
    plt.xlim(0, img_width)
    plt.ylim(0,img_height)  # Start the y-axis at the top to match image coordinates
    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Center of Words')
    plt.show()

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


def _compare_words_similarity(word1, word2, distance_margin):
    return l_distance(word1, word2) <= distance_margin

def _has_overlap(y1_min, y1_max, y2_min, y2_max, minimum_overlap=10):
    overlap_start = max(y1_min, y2_min)
    overlap_end = min(y1_max, y2_max)

    # If the segments overlap, return the overlap length, otherwise return 0
    return max(0, overlap_end - overlap_start - minimum_overlap)

def _get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin, verbose):
    # Find the bounding box for the search word
    key_box = None
    for box, word in bounding_boxes:
        if _compare_words_similarity(word.lower(), key_word.lower(), distance_margin):
            if verbose:
                print(word, key_word)
            key_box = box
            break
    if verbose:
        print(f'Key box defined to {key_box}')
    
    return key_box, key_word

def compute_box_distance(box_1, box_2, mode="euclidian"):
    """
    Compute the Euclidean distance between the centers of two boxes.

    Parameters:
    - key_box (tuple): A tuple representing the first box (key_x_min, key_x_max, key_y_min, key_y_max).
    - box (tuple): A tuple representing the second box (b_x_min, b_x_max, b_y_min, b_y_max).

    Returns:
    - float: The Euclidean distance between the centers of the two boxes.
    """
    if mode == "euclidian": 
        x_min_1, x_max_1, y_min_1, y_max_1 = box_1
        x_min_2, x_max_2, y_min_2, y_max_2  = box_2
        
        # Calculate the center coordinates of each box
        center_x_1 = (x_min_1 + x_max_1) / 2
        center_y_1 = (y_min_1 + y_max_1) / 2
        center_x_2 = (x_min_2 + x_max_2) / 2
        center_y_2 = (y_min_2 + y_max_2) / 2
        
        # Compute the Euclidean distance between the centers
        distance = ((center_x_1 - center_x_2) ** 2 + (center_y_1 - center_y_2) ** 2) ** 0.5
    elif mode == "right_horizontal": 
        x_max_2 - x_max_1  # Calculate distance right coordinates
    
    return distance



def find_next_right_word(bounding_boxes, key_word, distance_margin=2, max_distance=200, verbose=False):
    """
    Find the closest word to the right of a given key word based on bounding boxes.

    Parameters:
    - bounding_boxes (list): A list of tuples where each tuple contains a bounding box and a word. 
                             Each bounding box is represented as (x_min, x_max, y_min, y_max).
    - key_word (str): The reference word to which the closest right-side word is to be found.
    - distance_margin (int, optional): Margin to consider while matching the key word with words in bounding_boxes. 
                                       Defaults to 1.
    - max_distance (int, optional): Maximum allowed distance between key_word and the word to its right. 
                                    Defaults to 200.
    - verbose (bool, optional): If True, the function will print additional debug information. Defaults to False.

    Returns:
    - dict: A dictionary containing the bounding box and word for both the detected key word 
            and the closest word to its right. If no word is found to the right, the 'next' key will have None values.

    Example:
    ```
    bounding_boxes = [((10, 20, 10, 20), "hello"), ((25, 35, 10, 20), "world")]
    find_next_right_word(bounding_boxes, "hello")
    # Output: {'detected': ((10, 20, 10, 20), 'hello'), 'next': ((25, 35, 10, 20), 'world')}
    ```

    Note:
    The function considers words to be on the same line if their y-coordinates overlap.
    """
    key_box, key_word = _get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin,verbose)
    
    # If the word isn't found, return None
    if key_box is None:
        print("No box corresponding to the key word found")
        return "<NOT_FOUND>"
    
    # Variables to store the closest word and its distance
    found_value = False
    closest_distance = float('inf')
    key_x_min, key_x_max, key_y_min, key_y_max = key_box
    for box, word in bounding_boxes:
        # Check if the word's y-coordinates overlap with the key word's
        b_x_min, b_x_max, b_y_min, b_y_max = box
        overlap = _has_overlap(key_y_min, key_y_max, b_y_min, b_y_max,minimum_overlap=10)
        if overlap:
            if verbose:
                print(f'{word} overlaps {key_word}')
            # Check if the word is to the right of the key word
            if b_x_min > key_x_min: #There maybe x overlap as well
                
                distance = compute_box_distance(key_box,
                                               box,mode="euclidian")
                if verbose: 
                    print(f'distance to {word}',distance)
                if distance < min(closest_distance,max_distance):
                    closest_distance = distance
                    closest_word = word
                    closest_box = box
                    found_value = True

    if not found_value:
        return "<EMPTY>"
    
    return {'detected': (key_box, key_word),
            'next':(closest_box, closest_word)}

def merge_word_with_following(converted_boxes, key_word, verbose=False, safe=False):
    """
    Merge the bounding box and word of a given key_word with its closest following word.
    
    Parameters:
    - converted_boxes (list of tuple): A list of tuples, where each tuple contains a bounding box and a word.
                                      Each bounding box is represented as (x_min, x_max, y_min, y_max).
    - key_word (str): The word whose bounding box needs to be merged with the closest word to its right.
    - verbose (bool): If True, the function will print additional debug information.

    Returns:
    - list of tuple: An updated list of bounding boxes after merging the key_word with its following word.

    Note:
    If the key_word does not have a following word in the given list, the function will not make any changes.
    """
    try:
        data = find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=verbose)
        if verbose:
            print('data')
        
        new_word = data['detected'][1] + ' ' + data['next'][1]
        new_box = (min(data['detected'][0][0], data['next'][0][0]),
                   max(data['detected'][0][1], data['next'][0][1]),
                   min(data['detected'][0][2], data['next'][0][2]),
                   max(data['detected'][0][3], data['next'][0][3]))
        
        converted_boxes.remove(data['detected'])
        converted_boxes.remove(data['next'])
        converted_boxes.append((new_box, new_word))

    except Exception as e:
        if not safe:
            raise e

    return converted_boxes


# +
def run_doctr_model(img_path):
    ''' Load an image and run the model on it.'''
    img = DocumentFile.from_images(img_path)
    model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)
    result = model(img)
    output = result.export()
    return output
    
def get_processed_boxes_and_words(img_path, block, verbose=False):
    ''' Load an image, run the model, get the box words pairs and preprocess the needed words.'''
    output = run_doctr_model(img_path)
    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output, verbose)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    return converted_boxes

def postprocess_boxes_and_words(converted_boxes, block, safe, verbose):
    converted_boxes = remove_word(converted_boxes, ":")
    if block == 'block_2':
        converted_boxes = merge_word_with_following(converted_boxes,'Restitué', safe, verbose)
        converted_boxes = merge_word_with_following(converted_boxes,'Lieu', safe, verbose)
    if block == 'block_5':
        converted_boxes = merge_word_with_following(converted_boxes,'Nom', safe, verbose)

    return converted_boxes



# -

def remove_word(converted_boxes, 
                word):
    return [(b,w) for (b,w) in converted_boxes if w != word]


def plot_boxes_with_text(data):
    """
    Plots given bounding boxes and their associated text.
    
    Parameters:
    - data (list): A list of tuples, where each tuple contains bounding box coordinates and associated text.
    
    Example:
    data = [([128, 256, 221, 248], 'LEVEHICULE'), ...]
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    for box, word in data:
        x_min, x_max, y_min, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x_min, y_min, word, color='blue', verticalalignment='bottom')

    ax.set_ylim(0, 300)  # adjust these values if necessary
    ax.set_xlim(0, 1200)  # adjust these values if necessary
    plt.show()


# ## I. Running the detection & recognition model

# +
#img_path = "../arval/barth_jpg/arval_fleet_service_restitution/DY-984-XY_PV de reprise_p2.jpeg"
img_path = "data/test_data/DM-984-VT_Proces verbal de restitution_page-0001_bloc_2.png"
img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()
# -


# Get and print the blocs :
graphical_coordinates = get_block_coordinates(output)
image = PIL.Image.open(img_path)
result_image = draw_bounds(image, graphical_coordinates)
plt.figure(figsize=(15,15))
plt.imshow(result_image)

# ## II. Preprocessing 

# Getting all the words and converting their bounding box to cartesian coords (y from the bottom to the top).

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output,verbose=True)
image_dims = (output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])  # Replace with your image dimensions
print(image_dims)
converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
print(converted_boxes)

# Preprocessing of the key words composed of multiple words (for instance "Restitué" -> "Restitué le").

# +
img_path = "data/test_data/DM-984-VT_Proces verbal de restitution_page-0001_bloc_2.png"
img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output)


converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
merge_word_with_following(converted_boxes,'Restitué')
# -

# ## III. Getting the value for a given key

key_word = ""
print(find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=True))

plot_centers2(converted_boxes, image_dims)

# ## IV Performance estimation

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

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result_template = get_result_template()

all_results = []
for element in image_list:
    print(f'==== Running for file: {element} =====')
    filename_prefix = f'{element[:-5]}'
    result_json = {}
    result_json['File Name'] = element
    for bn in list(result_template.keys()):
        result_json[bn] = {}
        path_to_name_jpeg = FOLDER_IMAGES/element/'blocks'/(element + f"_{bn.replace('_',' ')}.png")
        converted_boxes = get_processed_boxes_and_words(img_path=path_to_name_jpeg,
                                                        block=bn,
                                                        verbose=VERBOSE)
        converted_boxes = postprocess_boxes_and_words(converted_boxes,
                                                      block=bn,
                                                      verbose=VERBOSE,
                                                      safe=True)
        for key_word in result_template[bn]:
            print(f'Running {key_word}')
            result_json[bn][key_word] = find_next_right_word(converted_boxes, key_word, 
                                                             distance_margin=2, verbose=VERBOSE)
            if has_found_box(result_json[bn][key_word]):
                result_json[bn][key_word] = result_json[bn][key_word]['next']
    all_results.append(result_json)


plot_boxes_with_text(converted_boxes)

pprint(all_results)

pprint(all_results[0])


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
# -

actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]

# +
from performance_estimation import compute_metrics_for_multiple_jsons


metrics = compute_metrics_for_multiple_jsons(predicted_dict_list, actual_json_list)
print(metrics)


# +
#TODO: Identify a way to distinguish the "le" from "le conducteur" and for the date.
