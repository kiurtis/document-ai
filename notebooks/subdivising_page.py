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
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import PIL
from PIL import ImageDraw
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance as l_distance
import math

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

def get_words_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    text_coordinates_and_word = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:                
                converted_coordinates = convert_coordinates(obj3["geometry"],page_dim)
                
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

def _has_overlap(y1_min, y1_max, y2_min, y2_max, minimum_overlap=5):
    return (y2_min  <= y1_min <= y2_max - minimum_overlap) or \
           (y2_min + minimum_overlap <= y1_max <= y2_max)

def _get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin, verbose):
    # Find the bounding box for the search word
    key_box = None
    for box, word in bounding_boxes:
        if _compare_words_similarity(word.lower(), key_word.lower(), distance_margin):
            print(word, key_word)
            key_box = box
            break
    if verbose:
        print(f'Key box defined to {key_box}')
    
    return key_box, key_word

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

def merge_word_with_following(converted_boxes, key_word):
    """
    Modified converted_boxes to remove the key_word/key_box tuple and the next_word/next_box tuple 
    and replace it by a tuple with concatenated key words and merged boxes (taking the maximum extent). 
    """
    data = find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=True)
    new_word = data['detected'][1] + ' ' + data['next'][1]
    new_box = (min(data['detected'][0][0],data['next'][0][0]),
               max(data['detected'][0][1],data['next'][0][1]),
               min(data['detected'][0][2],data['next'][0][2]),
               max(data['detected'][0][3],data['next'][0][3]))
    converted_boxes.pop(converted_boxes.index(data['detected']))
    converted_boxes.pop(converted_boxes.index(data['next']))
    converted_boxes.append((new_box,new_word))
    return converted_boxes


# -

# ## I. Running the detection & recognition model

# +
#img_path = "../arval/barth_jpg/arval_fleet_service_restitution/DY-984-XY_PV de reprise_p2.jpeg"
img_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/pop_files/myzip_jpg/arval_fleet_service_restitution/transposed_2/DM-984-VT_Proces_verbal_de_restitution_page-0001.jpg"
img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()
# -


# Get and print the blocs :
graphical_coordinates = get_block_coordinates(output)
image = PIL.Image.open(img_path)
result_image = draw_bounds(image, graphical_coordinates[4:5])
plt.figure(figsize=(15,15))
plt.imshow(result_image)

# ## II. Preprocessing 

# Getting all the words and converting their bounding box to cartesian coords (y from the bottom to the top).

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output)
image_dims = (output['pages'][0]["dimensions"][1], output['pages'][0]["dimensions"][0])  # Replace with your image dimensions
print(image_dims)
converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
print(converted_boxes)

converted_boxes

# Preprocessing of the key words composed of multiple words (for instance "Restitué" -> "Restitué le").

merge_word_with_following(converted_boxes,'Restitué')

# +
#img_path = "../test_data/DM-984-VT_Proces verbal de restitution_page-0001_bloc_2.png"
img_path = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/pop_files/myzip_jpg/arval_fleet_service_restitution/transposed_2/DM-984-VT_Proces_verbal_de_restitution_page-0001.jpg"

img = DocumentFile.from_images(img_path)

model = ocr_predictor(det_arch = 'db_resnet50',reco_arch = 'crnn_mobilenet_v3_large',pretrained = True)

result = model(img)
output = result.export()

graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output)


converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
merge_word_with_following(converted_boxes,'Restitué')


# +
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
    graphical_coordinates, text_coordinates_and_word = get_words_coordinates(output)
    converted_boxes = convert_to_cartesian(text_coordinates_and_word, image_dims)
    converted_boxes = merge_word_with_following(converted_boxes,'Restitué')
    return converted_boxes


# -

# ## III. Getting the value for a given key

key_word = ""
print(find_next_right_word(converted_boxes, key_word, distance_margin=1, verbose=True))

plot_centers2(converted_boxes, image_dims)
print(converted_boxes)
print(image_dims)


# ## IV Creating Separator

# +

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


# -

plot_boxes(converted_boxes, image_dims)
print(image_dims)
image_dims[1]


# +

def find_max_spacing_non_crossing_lines(bound, img_height, max_lines=1):
    non_crossing_lines = []

    # Define a function to check if a line crosses a bounding box
    def does_line_cross_box(y, box):
        _, _, y1, y2 = box
        return y >= y1 and y <= y2

    # Iterate through possible y-coordinates (lines)
    for y in range(img_height):
        crosses = False

        # Check if the line crosses any bounding boxes
        for b in bound:
            b1 = b[0]
            if does_line_cross_box(y, b1):
                crosses = True
                break  # No need to check other boxes if it crosses one

        # If the line doesn't cross any bounding boxes, add it to the list
        if not crosses:
            non_crossing_lines.append(y)

    # Sort the non-crossing lines based on their position
    non_crossing_lines.sort()

    # Calculate the available space between the lines
    available_space = [non_crossing_lines[i+1] - non_crossing_lines[i] for i in range(len(non_crossing_lines) - 1)]

    # Randomly select a number of lines up to the maximum specified
    num_lines = min(max_lines, len(non_crossing_lines))
    
    # Select lines that maximize the spacing between them
    selected_lines = []

    while len(selected_lines) < num_lines:
        # Find the index of the largest available space
        max_space_index = available_space.index(max(available_space))
        
        # Add the corresponding line to the selected lines
        selected_lines.append(non_crossing_lines[max_space_index + 1])
        
        # Update the available space list by removing the used space
        available_space.pop(max_space_index)

    return selected_lines




# Example usage:
img_height = image_dims[1]
non_crossing_lines = find_max_spacing_non_crossing_lines(converted_boxes, img_height,4)
print("Non-crossing lines:", non_crossing_lines)

# +
import matplotlib.pyplot as plt
import random  # For generating random line colors



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

# Example usage:

plot_boxes_and_lines(converted_boxes, image_dims,non_crossing_lines)
#print(image_dims)

# +
import cv2
import os


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
            y2 = selected_lines[i + 1]
        else:
            y2 = image.shape[0]  # Use the image height for the last segment

        # Crop the image
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


# -

img_path
non_crossing_lines
sorted_lines = sorted(non_crossing_lines)
print(sorted_lines)

output_folder = "/Users/pipobimbo/Desktop/Caliente_work/OCR_test/document-ai_barth/document-ai/notebooks/test_out"

output_folder

cut_and_save_image(img_path, output_folder, sorted_lines)


