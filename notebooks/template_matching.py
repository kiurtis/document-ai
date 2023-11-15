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
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
    
from PIL import Image
import matplotlib.pyplot as plt


# Load the image
image_path = 'data/performances_data/arval_classic_restitution_images/DH-427-VH_PV_RESTITUTION_DH-427-VH_p1.jpeg'
image = Image.open(image_path)



# Display the image to estimate the coordinates

plt.imshow(image)

plt.show()

# +
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to perform template matching at multiple scales
def multi_scale_template_matching(image_path, template_path):
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

    # Display the result
    plt.imshow(img_rgb)
    plt.title(f'Scale: {best_scale} - Quality: {best_match_quality}')
    plt.show()

    return output_image_path, best_match_quality, best_scale

# Define the paths to your image and template
image_path = 'data/performances_data/arval_classic_restitution_images/DH-427-VH_PV_RESTITUTION_DH-427-VH_p1.jpeg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_top_left.png'

# Call the function with your image paths
output_image_path, match_quality, scale = multi_scale_template_matching(image_path, template_path)
print(f"Output Image Path: {output_image_path}")
print(f"Match Quality: {match_quality}")
print(f"Scale: {scale}")


# +
import cv2
import numpy as np
import os
from pathlib import Path

# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
output_directory = Path('data/performances_data/arval_classic_restitution_images/croped/')
image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print(f"Processecing {file_path}")

        output_image_path, match_quality, scale = multi_scale_template_matching(file_path, template_path)
        print(f"Output Image Path: {output_image_path}")
        print(f"Match Quality: {match_quality}")
        print(f"Scale: {scale}")
        
        print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")

print("All files have been processed.")


# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path = 'data/performances_data/arval_classic_restitution_images/template/bottom_right.png'
image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print(f"Processecing {file_path}")

        output_image_path, match_quality, scale = multi_scale_template_matching(file_path, template_path)
        print(f"Output Image Path: {output_image_path}")
        print(f"Match Quality: {match_quality}")
        print(f"Scale: {scale}")
        
        print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")

print("All files have been processed.")

# -

# # II Trying to find several iterations of a template

# +
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




# +
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


image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'

top_left = (502, 321)
bottom_right = (838, 367)



# Call the function to remove the rectangle from the image

output_image_path = remove_rectangle_from_image(image_path, output_path ,top_left, bottom_right)
print(output_image_path)


# -

def multi_scale_multi_results_temp_matching(image_path, temp_path, number_of_iteration,print_img=False,print2=False):
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
    


# +
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


# Call the function to contour rectangles on the image
#contour_rectangles_on_image(image_path, list_coord)

#output_image_path

# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
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
        
        list_coord = multi_scale_multi_results_temp_matching(file_path,template_path,3,print_img=False,print2=True)
        
        print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")
        
    

print("All files have been processed.")
# -

image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)

# +
import numpy as np

def calculate_distance_between_rectangles(rectangles):

    # Calculate the center of the first rectangle
    center_1 = ((rectangles[0][0][0] + rectangles[0][1][0]) / 2, (rectangles[0][0][1] + rectangles[0][1][1]) / 2)
    # Calculate the center of the second rectangle
    center_2 = ((rectangles[1][0][0] + rectangles[1][1][0]) / 2, (rectangles[1][0][1] + rectangles[1][1][1]) / 2)

    # Compute the Euclidean distance between the two centers
    distance = np.sqrt((center_1[0] - center_2[0])**2 + (center_1[1] - center_2[1])**2)
    return distance


def compute_arval_bar_code_distance(image_path,temp_path_arv,temp_path_code_bar,print_result=False):
    list_coord =[]
    output_image_path, match_quality, scale, best_match_coordinates =multi_scale_template_matching2(image_path,temp_path_arv)
    list_coord.append(best_match_coordinates)
    output_image_path, match_quality, scale, best_match_coordinates =multi_scale_template_matching2(image_path,temp_path_code_bar)
    list_coord.append(best_match_coordinates)

    distance = calculate_distance_between_rectangles(list_coord)
    if print_result:
        contour_rectangles_on_image(image_path, list_coord)
    return list_coord, distance



# +
image_path = 'data/performances_data/arval_classic_restitution_images/EL-935-PX_EL-935-PX_Pv_de_restitution_p1.jpeg'
temp_path_code_bar = 'data/performances_data/arval_classic_restitution_images/template/bottom_right.png'
temp_path_arv = 'data/performances_data/arval_classic_restitution_images/template/template_top_left.png'

list_coord,distance = compute_arval_bar_code_distance(image_path,temp_path_arv,temp_path_code_bar,print_result=True)
print(list_coord,distance)

# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
temp_path_code_bar = 'data/performances_data/arval_classic_restitution_images/template/bottom_right.png'
temp_path_arv = 'data/performances_data/arval_classic_restitution_images/template/template_top_left.png'
image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

res = []
# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print(f"Processing {file_path}")
        
        list_coord,distance = compute_arval_bar_code_distance(file_path,temp_path_arv,temp_path_code_bar,print_result=False)
        print(list_coord,distance)        
        print(f"Processed {file_path}")
        res.append((file_path,distance))
    else:
        print(f"Skipped non-image file: {file_name}")
        
    

print("All files have been processed.")
# -

# # SElecting the right boxes

# +
from PIL import Image

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns a tuple (width, height)



# -

image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)
dimensions = get_image_dimensions(image_path)
print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")


list2 = [((502, 321), (838, 367)), ((1047, 2829), (1383, 2875))]
contour_rectangles_on_image(image_path, list2)

# +
import math

def calculate_relative_metrics(rect, doc_width, doc_height):
    ((x1, y1), (x2, y2)) = rect
    rel_x1, rel_y1 = x1 / doc_width, y1 / doc_height
    rel_width, rel_height = (x2 - x1) / doc_width, (y2 - y1) / doc_height
    return (rel_x1, rel_y1, rel_width, rel_height)

def is_within_tolerance(rect_a, rect_b, tolerance=0.08):
    for a, b in zip(rect_a, rect_b):
        if abs(a - b) > tolerance:
            return False
    return True

def select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics):
    # Calculate relative metrics for each box
    relative_metrics = [calculate_relative_metrics(box, current_doc_width, current_doc_height) for box in boxes]

    # Identify boxes that match the desired relative metrics
    selected_boxes = []

    for box, metrics in zip(boxes, relative_metrics):
        for target in target_relative_metrics:
            if is_within_tolerance(metrics, target):
                selected_boxes.append(box)
                break

    return selected_boxes

# Example usage
boxes =[((502, 321), (838, 367)), ((1047, 2829), (1383, 2875)), ((351, 3368), (510, 3390))]  # Replace with your box coordinates in the new document
current_doc_width, current_doc_height = 2480, 3509   # Replace with the dimensions of the new document

# Target relative metrics from the reference document
target_relative_metrics = [
    (0.2024, 0.0915, 0.1355, 0.0131),
    (0.4222, 0.8062, 0.1355, 0.0131)
]
target_relative_metrics2= [calculate_relative_metrics(((502, 321), (838, 367)), 2480, 3509),calculate_relative_metrics(((1047, 2829), (1383, 2875)), 2480, 3509)]


selected_boxes = select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics2)
print("Selected Rectangles:", selected_boxes)

#print(target_relative_metrics2)

contour_rectangles_on_image(image_path, selected_boxes)

# -

image_path = 'data/performances_data/arval_classic_restitution_images/EF-988-TA_procès_restitution_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)
dimensions = get_image_dimensions(image_path)
print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")

# +
boxes = list_coord
current_doc_width = dimensions[0]
current_doc_height = dimensions[1]

selected_boxes = select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics2)
print("Selected Rectangles:", selected_boxes)

#print(target_relative_metrics2)

contour_rectangles_on_image(image_path, selected_boxes)

# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
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
        
        list_coord = multi_scale_multi_results_temp_matching(file_path,template_path,3,print_img=False,print2=True)
        print('list_coord 3 rect =',list_coord)
        #Select the right rectangles
        dimensions = get_image_dimensions(file_path)
        current_doc_width = dimensions[0]
        current_doc_height = dimensions[1]
        
        selected_boxes = select_boxes(list_coord, current_doc_width, current_doc_height, target_relative_metrics2)
        print("Selected Rectangles:", selected_boxes)

        contour_rectangles_on_image(file_path, selected_boxes)

        
        #print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")
        
    

print("All files have been processed.")

# +
image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)
dimensions = get_image_dimensions(image_path)
ref_doc_dim = get_image_dimensions(image_path)
print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")

boxes = list_coord
current_doc_width = dimensions[0]
current_doc_height = dimensions[1]

selected_boxes = select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics2)
print("Selected Rectangles:", selected_boxes)

#print(target_relative_metrics2)

contour_rectangles_on_image(image_path, selected_boxes)

# +
image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'

search_rectangles = [((0, 767), (2480, 1180)),((1600, 767), (2320, 1180)), ((0, 2829), (2480, 3250)), ((120, 2829), (1600, 3250))]

contour_rectangles_on_image(image_path, [((0, 767), (2480, 1180))])
print(dimensions)


# +
def find_block5(ref_rect, ref_size, ref_block5, new_size, new_ref_rect):
    """
    Calculate the coordinates for block5 on a new document based on reference rectangle and document sizes.

    Parameters:
    ref_rect (tuple): The reference rectangle coordinates on the reference document.
    ref_size (tuple): The width and height of the reference document.
    ref_block5 (tuple): The block5 coordinates on the reference document.
    new_size (tuple): The width and height of the new document.
    new_ref_rect (tuple): The reference rectangle coordinates on the new document.

    Returns:
    tuple: The coordinates for the new block5 on the new document.
    """

    # The top of the new block5 is the same as the top of the new reference rectangle
    new_block5_top = new_ref_rect[0][1]

    # Calculate the height proportion of the block5 relative to the reference document height
    ref_block5_height = ref_block5[1][1] - ref_block5[0][1]
    height_proportion = ref_block5_height / ref_size[1]

    # Apply the height proportion to the new document height to get the new block5 height
    new_block5_height = int(height_proportion * new_size[1])

    # The bottom of the new block5 is the top plus the new proportional height
    new_block5_bottom = new_block5_top + new_block5_height

    # The width of the new block5 is the same as the width of the new document
    new_block5_left = 0
    new_block5_right = new_size[0]

    # Return the new block5 coordinates
    return ((new_block5_left, new_block5_top), (new_block5_right, new_block5_bottom))

def select_bottom_rectangle(rectangles):
    """
    Selects the bottom rectangle from a list of rectangles.

    Parameters:
    rectangles (list of tuples): A list of rectangle coordinates, where each rectangle is defined by top-left and bottom-right coordinates.

    Returns:
    tuple: The coordinates of the bottom rectangle.
    """
    
    # Sort the rectangles by the y-coordinate of the top-left corner and then by the y-coordinate of the bottom-right corner
    # The bottom rectangle will have the highest y-coordinate at the top-left corner
    sorted_rectangles = sorted(rectangles, key=lambda x: (x[0][1], x[1][1]), reverse=True)

    # Return the first rectangle in the sorted list, which is the bottom rectangle
    return sorted_rectangles[0]

def calculate_new_block2(ref_rectangles, ref_block2, ref_dimensions, new_dimensions, new_ref_rectangles):
    """
    Calculates the coordinates of block 2 on a new document based on the reference rectangles and dimensions.
    
    Parameters:
    ref_rectangles (list of tuple): Reference rectangles on the original document.
    ref_block2 (tuple): Reference block 2 coordinates on the original document.
    ref_dimensions (tuple): Dimensions of the original document (width, height).
    new_dimensions (tuple): Dimensions of the new document (width, height).
    new_ref_rectangles (list of tuple): Reference rectangles on the new document.
    Returns:
    tuple: The coordinates for the new block 2 on the new document.
    """

    # Calculate the vertical scale factor using the distance between the reference rectangles on the reference document
    ref_distance = ref_rectangles[1][0][1] - ref_rectangles[0][1][1]
    new_distance = new_ref_rectangles[1][0][1] - new_ref_rectangles[0][1][1]
    vertical_scale_factor = new_distance / ref_distance

    # Calculate the vertical position of the new block 2 based on the scale factor and the new reference rectangles
    new_block2_top_y = new_ref_rectangles[0][1][1] + int((ref_block2[0][1] - ref_rectangles[0][1][1]) * vertical_scale_factor)
    new_block2_bottom_y = new_ref_rectangles[0][1][1] + int((ref_block2[1][1] - ref_rectangles[0][1][1]) * vertical_scale_factor)
 
    # The width of block 2 is the same as the width of the new document
    new_block2_left_x = 0
    new_block2_right_x = new_dimensions[0]

    # Return the new block 2 coordinates
    return ((new_block2_left_x, new_block2_top_y), (new_block2_right_x, new_block2_bottom_y))



# Example usage:
    


ref_block2 = ((0, 767), (2480, 1180))

# Example usage:
ref_rect_bot = ((1047, 2829), (1383, 2875)) 
ref_size = ref_doc_dim
ref_block5 = ((0, 2829), (2480, 3250))

ref_rects = [((502, 321), (838, 367)), ((1047, 2829), (1383, 2875))]


contour_rectangles_on_image(image_path, ref_rects)


# +

image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)
dimensions = get_image_dimensions(image_path)
ref_doc_dim = get_image_dimensions(image_path)
print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")

boxes = list_coord
current_doc_width = dimensions[0]
current_doc_height = dimensions[1]

selected_boxes = select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics2)
print("Selected Rectangles:", selected_boxes)

#print(target_relative_metrics2)

contour_rectangles_on_image(image_path, selected_boxes)


# Placeholder for new document's dimensions and reference rectangles
# You will need to replace these placeholders with actual values
new_doc_size = dimensions # Example new document height
#new_ref_rectangle = selected_boxes[1] # Example new reference rectangles
sorted_rectangles = sorted(selected_boxes, key=lambda x: (x[0][1], x[1][1]))
new_ref_rectangle_bot = sorted_rectangles[1]
print(new_ref_rectangle)

print(ref_doc_dim)
print(new_doc_size)


# Calculate the new block5
new_block5 = find_block5(ref_rect_bot, ref_size, ref_block5, new_doc_size, new_ref_rectangle_bot)

contour_rectangles_on_image(image_path, [new_block5,ref_rect])

new_block2 = calculate_new_block2(ref_rectangles, ref_block2, ref_size, new_doc_size, sorted_rectangles)

contour_rectangles_on_image(image_path, [new_block2])


# +
#Test on new Doc:

image_path = 'data/performances_data/arval_classic_restitution_images/EF-988-TA_procès_restitution_p1.jpeg'

template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,3,print_img=False,print2=True)
print(list_coord)
dimensions = get_image_dimensions(image_path)
print(f"Width: {dimensions[0]}, Height: {dimensions[1]}")

boxes = list_coord
current_doc_width = dimensions[0]
current_doc_height = dimensions[1]

selected_boxes = select_boxes(boxes, current_doc_width, current_doc_height, target_relative_metrics2)
print("Selected Rectangles:", selected_boxes)

#print(target_relative_metrics2)

contour_rectangles_on_image(image_path, selected_boxes)

# +
new_doc_size = dimensions # 
new_ref_rectangle = selected_boxes[1] # Example new reference rectangles

sorted_rectangles = sorted(selected_boxes, key=lambda x: (x[0][1], x[1][1]))
new_ref_rectangle_bot = sorted_rectangles[1]

print(ref_doc_dim)
print(new_doc_size)
# Calculate the search rectangles for the new document using actual new dimensions and reference rectangles
new_block5 = find_block5(ref_rect_bot, ref_size, ref_block5, new_doc_size, new_ref_rectangle_bot)

contour_rectangles_on_image(image_path, [new_block5,new_ref_rectangle])

new_block2 = calculate_new_block2(ref_rectangles, ref_block2, ref_size, new_doc_size, sorted_rectangles)

contour_rectangles_on_image(image_path, [new_block2])


# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
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
        
        list_coord = multi_scale_multi_results_temp_matching(file_path,template_path,3,print_img=False,print2=False)
        print('list_coord 3 rect =',list_coord)
        #Select the right rectangles
        dimensions = get_image_dimensions(file_path)
        current_doc_width = dimensions[0]
        current_doc_height = dimensions[1]
        
        selected_boxes = select_boxes(list_coord, current_doc_width, current_doc_height, target_relative_metrics2)
        print("Selected Rectangles:", selected_boxes)

        contour_rectangles_on_image(file_path, selected_boxes)

        new_doc_size = dimensions # 

        #new_ref_rectangle = select_bottom_rectangle(selected_boxes) 
        sorted_rectangles = sorted(selected_boxes, key=lambda x: (x[0][1], x[1][1]))
        new_ref_rectangle_bot = sorted_rectangles[1]

        
        print(ref_doc_dim)
        print(new_doc_size)
        # Calculate the search rectangles for the new document using actual new dimensions and reference rectangles
        new_block5 = find_block5(ref_rect, ref_size, ref_block5, new_doc_size, new_ref_rectangle_bot)
        new_block2 = calculate_new_block2(ref_rectangles, ref_block2, ref_size, new_doc_size, sorted_rectangles)


        contour_rectangles_on_image(file_path, [new_block5,new_block2])

        
        #print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")
        
    

print("All files have been processed.")
# -

# # Test doing the same with vehicule

# +
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
    



# +
from PIL import Image
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
from pathlib import Path

# !pwd

# +


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

def create_block5_from_ref_rectangles(top_rect, bottom_rect, page_width):
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
    block5 = ((0, top_rect[1][1]), (page_width, bottom_rect[0][1]))
    
    return block5


# +
# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')
template_path_le_vehi = 'data/performances_data/arval_classic_restitution_images/template/temp_le_vehicule_bis.png'
template_path_du_vehi = 'data/performances_data/arval_classic_restitution_images/template/temp_descriptif.png'
template_path_du_gros = 'data/performances_data/arval_classic_restitution_images/template/gros_schema.png'
template_path_du_code = 'data/performances_data/arval_classic_restitution_images/template/bottom_right.png'
image_files = os.listdir(image_directory)
print(image_files)


print(template_path_du_vehi)
# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print(f"Processing {file_path}")
        
        list_coord_top = multi_scale_multi_results_temp_matching(file_path,template_path_le_vehi,1,print_img=False,print2=False)
        print(list_coord_top)
        list_coord_botom = multi_scale_multi_results_temp_matching(file_path,template_path_du_vehi,1,print_img=False,print2=False)
        print(list_coord_botom)
        list_coord_schema = multi_scale_multi_results_temp_matching(file_path,template_path_du_gros,1,print_img=False,print2=False)
        print(list_coord_schema)
        list_coord_codebar = multi_scale_multi_results_temp_matching(file_path,template_path_du_code,1,print_img=False,print2=False)
        print(list_coord_codebar)
        
        new_dimensions = get_image_dimensions(file_path)
        print(new_dimensions)
        current_doc_width = new_dimensions[0]
        current_doc_height = new_dimensions[1]

        bloc2 = create_block2_from_ref_rectangles(list_coord_top[0], list_coord_botom[0], current_doc_width)
        print(bloc2)
        print([list_coord_top,list_coord_botom])
        bloc5 = create_block5_from_ref_rectangles(list_coord_schema[0], list_coord_codebar[0], current_doc_width)
        contour_rectangles_on_image(file_path, [bloc2,bloc5])
        print(f"Processed {file_path}")
    else:
        print(f"Skipped non-image file: {file_name}")
        
    

print("All files have been processed.")
# -


