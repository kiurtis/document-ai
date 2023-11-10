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
            output_image_path, match_quality, scale,best_match_coordinates =multi_scale_template_matching2(image_temp, template_path,print_img)
        except Exception as e:
            print("No match found or an error occurred:", e)
        list_coord.append(best_match_coordinates)
        top_left, bottom_right = best_match_coordinates[0], best_match_coordinates[1]
        image_temp = remove_rectangle_from_image(image_temp, output_path ,top_left, bottom_right,print_img)
    if print2:
        contour_rectangles_on_image(image_path,list_coord)
    return list_coord
    


print(list_coord)


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
contour_rectangles_on_image(image_path, list_coord)

output_image_path
# -

image_path = 'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg'
#output_path = 'data/performances_data/arval_classic_restitution_images/temp.jpg'
template_path = 'data/performances_data/arval_classic_restitution_images/template/template_conducteur.png'
list_coord = multi_scale_multi_results_temp_matching(image_path,template_path,4,print_img=False,print2=True)


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


