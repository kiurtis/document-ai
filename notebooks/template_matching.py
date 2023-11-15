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

# # Template matching functions to crop the file 

# +
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
    
# -

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from pathlib import Path


# +

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns a tuple (width, height)

# Function to perform template matching at multiple scales
def multi_scale_template_matching2(image_path, template_path,print_img=False):
    img = cv2.imread(image_path, 0)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    # List to store the results
    results = []

    # Initial scale range
    scales = np.linspace(0.5, 1.5, 20)

    # Perform initial template matching
    for scale in scales:
        resized_template = cv2.resize(template, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        w_resized, h_resized = resized_template.shape[::-1]
        
        res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        results.append((max_val, (max_loc, (max_loc[0] + w_resized, max_loc[1] + h_resized)), scale))

    # Sort the results by match quality
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    best_match_quality, best_match_coordinates, best_scale = sorted_results[0]

    # Continue with smaller scales if needed
    start_scale = best_scale - 0.05
    end_scale = 0.1
    for scale in np.linspace(start_scale, end_scale, 20):
        resized_template = cv2.resize(template, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        w_resized, h_resized = resized_template.shape[::-1]
        
        res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        results.append((max_val, (max_loc, (max_loc[0] + w_resized, max_loc[1] + h_resized)), scale))

    # Combine and sort all results
    all_sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    best_match_quality, best_match_coordinates, best_scale = all_sorted_results[0]

    # Draw a rectangle around the matched region
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img_rgb, best_match_coordinates[0], best_match_coordinates[1], (0, 255, 0), 2)

    # Save the result image
    output_image_path = 'match_result_updated.png'
    cv2.imwrite(output_image_path, img_rgb)
    if print_img:
        # Display the result
        plt.imshow(img_rgb)
        plt.title(f'Scale: {best_scale} - Quality: {best_match_quality}')
        plt.show()

    return output_image_path, best_match_quality, best_scale,best_match_coordinates


def remove_rectangle_from_image(image_path,output_path,top_left, bottom_right,print_img=False):

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not load the image. Please check the path.")

    # Convert coordinates to integer
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Set the ROI (Region of Interest) to black
    img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 0

    # Save the result
    output_image_path = output_path
    cv2.imwrite(output_image_path, img)

    if print_img:
        # Display the result
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Image with Rectangle Removed')
        plt.axis('off')
        plt.show()

    return output_image_path



def multi_scale_multi_results_temp_matching(image_path, temp_path,output_path, number_of_iteration,print_img=False,print2=False):
    list_coord = []
    image_temp = image_path
    for i in range(number_of_iteration):
        try:
            output_image_path, match_quality, scale,best_match_coordinates =multi_scale_template_matching2(image_temp, temp_path,print_img)
        except Exception as e:
            print("No match found or an error occurred:", e)
        list_coord.append(best_match_coordinates)
        top_left, bottom_right = best_match_coordinates[0], best_match_coordinates[1]
        image_temp = remove_rectangle_from_image(image_temp, output_path ,top_left, bottom_right,print_img)
    if print2:
        contour_rectangles_on_image(image_path,list_coord)
    return list_coord
    

def create_block2_from_ref_rectangles(top_rect, bottom_rect, page_width):
    """
    Defines block2 based on two reference rectangles. Block2 spans the full width of the document
    and vertically extends from the top of the top rectangle to the top of the bottom rectangle.

    Parameters:
    top_rect (tuple): The top-left and bottom-right coordinates of the top reference rectangle.
    bottom_rect (tuple): The top-left and bottom-right coordinates of the bottom reference rectangle.
    page_width (int): The full width of the document page.

    Returns:
    tuple: The coordinates of the block2 rectangle.
    """
    
    # Block2 starts at the top of the top rectangle and ends at the top of the bottom rectangle
    block2 = ((0, top_rect[0][1]), (page_width, bottom_rect[0][1]))
    
    return block2

def create_block4_from_ref_rectangles(top_rect, bottom_rect, page_width):

    block4 = ((0, top_rect[1][1]), (page_width, bottom_rect[0][1]))
    
    return block4

def contour_rectangles_on_image(image_path, rectangles):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load the image. Please check the path.")

    # Draw rectangles on the image
    for (top_left, bottom_right) in rectangles:
        # Convert tuple coordinates to integer
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        
        # Draw a rectangle on the image
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

    # Display the result
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Image with Rectangles Contoured')
    plt.axis('off')
    plt.show()

    return #output_image_path


# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path_top_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_le_vehicule.png'
template_path_top_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_descriptif.png'
template_path_bot_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_end_block3.png'
template_path_bot_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_barcode.png'

output_path = 'data/performances_data/arval_classic_restitution_images/output_cropped_images/temp.jpg'
image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print(f"Processing {file_path}")
        
        list_coord_top_bloc2 = multi_scale_multi_results_temp_matching(file_path,template_path_top_bloc2,output_path,1,print_img=False,print2=False)
        #print(list_coord_top)
        list_coord_top_bloc3 = multi_scale_multi_results_temp_matching(file_path,template_path_top_bloc3,output_path,1,print_img=False,print2=False)
        #print(list_coord_botom)
        list_coord_bot_bloc3 = multi_scale_multi_results_temp_matching(file_path,template_path_bot_bloc3,output_path,1,print_img=False,print2=False)
        #print(list_coord_schema)
        list_coord_bot_bloc4 = multi_scale_multi_results_temp_matching(file_path,template_path_bot_bloc4,output_path,1,print_img=False,print2=False)
        #print(list_coord_codebar)
        
        new_dimensions = get_image_dimensions(file_path)
        #print(new_dimensions)
        current_doc_width = new_dimensions[0]
        current_doc_height = new_dimensions[1]

        bloc2 = create_block2_from_ref_rectangles(list_coord_top_bloc2[0], list_coord_top_bloc3[0], current_doc_width)
        bloc4 = create_block4_from_ref_rectangles(list_coord_bot_bloc3[0], list_coord_bot_bloc4[0], current_doc_width)
        contour_rectangles_on_image(file_path, [bloc2,bloc4])
        print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")
        
    
    
print("All files have been processed.")
# -


