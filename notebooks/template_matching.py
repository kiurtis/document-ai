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

# # Template matching functions to crop the file into bloc 2 and bloc 4 and divide the info and the signature parts
#

# +
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
    

# +
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import matplotlib.image as mpimg

from ai_documents.detection.template_matching_function import get_image_dimensions,find_block2_infos,find_block4_infos,draw_contour_rectangles_on_image,crop_blocks_in_image,arval_classic_divide_and_crop_bloc2,arval_classic_divide_and_crop_bloc4


# +

# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')

template_path_top_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_le_vehicule.png'
template_path_top_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_descriptif.png'
template_path_bot_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_end_block3.png'
template_path_bot_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_barcode.png'

#template to subdivise the bloc:
template_path_signature_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_2_garage.png'
template_path_signature_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_4_long.png'

output_fold = '"/notebooks/Test_signa/"'

image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print('**************************************')
        print(f"Processing {file_path}")
        print('**************************************')

        try:
            # Getting bloc 2 and 4
            new_dimensions = get_image_dimensions(file_path)
            bloc2 = find_block2_infos(file_path, template_path_top_bloc2, template_path_top_bloc3, new_dimensions)
            bloc4 = find_block4_infos(file_path, template_path_bot_bloc3, template_path_bot_bloc4, new_dimensions)
            draw_contour_rectangles_on_image(file_path, [bloc2, bloc4])
            blocs = [bloc2, bloc4]
            print(blocs)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to get bloc 2 and 4 of ", file_name ," :", e)
            print('-----------------')
        
        
        image = cv2.imread(file_path)
        try:
            #cropping the image in bloc
            crop_blocks_in_image(image, blocs, output_fold, file_name)
            cropped_image_paths = [os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_{i}.jpeg") for i in range(len(blocs))]
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to crop the image ", file_name ," :", e)
            print('-----------------')

        #Dividing and cropping bloc 2:
        try:
            file_path_bloc2 = str(cropped_image_paths[0])
            bloc_2_info_path,bloc_2_sign_path = arval_classic_divide_and_crop_bloc2(file_path_bloc2,output_fold,file_name,template_path_signature_bloc2)

        
            image_path = bloc_2_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_2_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 2 in two", file_name ," :", e)
            print('-----------------')
        

        #Dividing and cropping bloc 4:
        try:
            file_path_bloc4 = str(cropped_image_paths[1])
            bloc_4_info_path,bloc_4_sign_path = arval_classic_divide_and_crop_bloc4(file_path_bloc4,output_fold,file_name,template_path_signature_bloc4)

        
            image_path = bloc_4_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_4_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 4 in two", file_name ," :", e)
            print('-----------------')
        

        
        #print(f"Processed {file_path}")
        print('-----------------------------------')
        print('   ')
        print('   ')
    else:
        print(f"Skipped non-image file: {file_name}")

    
    break
    
print("All files have been processed.")
# -

# # TESTING Signature detection

# +

def place_image_in_center(input_image_path, output_image_size, output_image_path):
    # Load the input image
    input_image = Image.open(input_image_path)

    # Create a new image with the specified dimensions
    output_image = Image.new("RGB", output_image_size, (255, 255, 255))  # White background

    # Calculate the position to place the input image so it is centered
    x = (output_image_size[0] - input_image.width) // 2
    y = (output_image_size[1] - input_image.height) // 2

    # Paste the input image onto the larger image
    output_image.paste(input_image, (x, y))

    # Save the resulting image
    output_image.save(output_image_path)

# Example usage
name="bolos"
s = "/notebooks/Test_signa/{}.".format(name)
print(s)
#place_image_in_center("path_to_your_input_image.jpg", (800, 600), "/notebooks/Test_signa/{}path_to_your_output_image.jpg")


# +

# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')

template_path_top_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_le_vehicule.png'
template_path_top_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_descriptif.png'
template_path_bot_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_end_block3.png'
template_path_bot_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_barcode.png'

#template to subdivise the bloc:
template_path_signature_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_2_garage.png'
template_path_signature_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_4_long.png'

output_fold = "data/performances_data/arval_classic_restitution_images/test_signa/"

image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print('**************************************')
        print(f"Processing {file_path}")
        print('**************************************')

        try:
            # Getting bloc 2 and 4
            new_dimensions = get_image_dimensions(file_path)
            bloc2 = find_block2_infos(file_path, template_path_top_bloc2, template_path_top_bloc3, new_dimensions)
            bloc4 = find_block4_infos(file_path, template_path_bot_bloc3, template_path_bot_bloc4, new_dimensions)
            draw_contour_rectangles_on_image(file_path, [bloc2, bloc4])
            blocs = [bloc2, bloc4]
            print(blocs)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to get bloc 2 and 4 of ", file_name ," :", e)
            print('-----------------')
        
        
        image = cv2.imread(file_path)
        try:
            #cropping the image in bloc
            crop_blocks_in_image(image, blocs, output_fold, file_name)
            cropped_image_paths = [os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_{i}.jpeg") for i in range(len(blocs))]
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to crop the image ", file_name ," :", e)
            print('-----------------')

        #Dividing and cropping bloc 2:
        try:
            file_path_bloc2 = str(cropped_image_paths[0])
            bloc_2_info_path,bloc_2_sign_path = arval_classic_divide_and_crop_bloc2(file_path_bloc2,output_fold,file_name,template_path_signature_bloc2)

        
            image_path = bloc_2_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_2_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
            
            out_croped_bloc2_path = output_fold +'bloc2_'+file_name
            print(out_croped_bloc2_path)
            print(bloc_2_sign_path)
            place_image_in_center(bloc_2_sign_path, new_dimensions, out_croped_bloc2_path)

        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 2 in two", file_name ," :", e)
            print('-----------------')
        

        #Dividing and cropping bloc 4:
        try:
            file_path_bloc4 = str(cropped_image_paths[1])
            bloc_4_info_path,bloc_4_sign_path = arval_classic_divide_and_crop_bloc4(file_path_bloc4,output_fold,file_name,template_path_signature_bloc4)

        
            image_path = bloc_4_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_4_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 4 in two", file_name ," :", e)
            print('-----------------')
        

        
        #print(f"Processed {file_path}")
        print('-----------------------------------')
        print('   ')
        print('   ')
    else:
        print(f"Skipped non-image file: {file_name}")

    
    #break
    
print("All files have been processed.")
# -

# # Checking if there is a signature/ stamp

# +


# Function to perform template matching at multiple scales
def multi_scale_template_matching2(image_path, template_path, print_img=False):

    def apply_template_matching(img, template, scale, results):
        resized_template = cv2.resize(template, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        w_resized, h_resized = resized_template.shape[::-1]

        res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        results.append((max_val, (max_loc, (max_loc[0] + w_resized, max_loc[1] + h_resized)), scale))

        return results

    def get_best_match(img, template, scales, results):
        # Perform initial template matching
        
        for scale in scales:
            results = apply_template_matching(img, template, scale, results)

        # Sort the results by match quality
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        best_match_quality, best_match_coordinates, best_scale = sorted_results[0]
        return best_match_quality, best_match_coordinates, best_scale

    img = cv2.imread(image_path, 0)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]

    # List to store the results
    results = []

    # Initial scale range
    scales = np.linspace(0.5, 0.6, 20)
    best_match_quality, best_match_coordinates, best_scale = get_best_match(img, template, scales, results)

    # Continue with smaller scales if needed
    start_scale = best_scale - 0.05
    end_scale = 0.2
    smaller_scales = np.linspace(start_scale, end_scale, 20)
    best_match_quality, best_match_coordinates, best_scale = get_best_match(img, template, smaller_scales, results)

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

def crop_bloc2_signature(file_path_bloc2, output_fold, file_name, template_path_signature_bloc2_top,template_path_signature_bloc2_bottom,template_path_signature_bloc2_right) :

        output_image_path, match_quality, scale, list_coord_signa_bloc2_1 = multi_scale_template_matching2(file_path_bloc2,
                                                                                                        template_path_signature_bloc2_top,
                                                                                                        print_img=False)

        output_image_path, match_quality, scale, list_coord_signa_bloc2_2 = multi_scale_template_matching2(file_path_bloc2,
                                                                                                        template_path_signature_bloc2_bottom,
                                                                                                        print_img=False)

        output_image_path, match_quality, scale, list_coord_signa_bloc2_3 = multi_scale_template_matching2(file_path_bloc2,
                                                                                                        template_path_signature_bloc2_right,
                                                                                                        print_img=False)

        blocs_signbloc2 = [list_coord_signa_bloc2_1,list_coord_signa_bloc2_2,list_coord_signa_bloc2_3]
        #print(blocs_signbloc2)
        inner_rectangle = find_custom_rectangle(blocs_signbloc2)
        print(inner_rectangle)
    
        #draw_contour_rectangles_on_image(file_path_bloc2, blocs_signbloc2)
        draw_contour_rectangles_on_image(file_path_bloc2, [inner_rectangle])

        is_uniform = check_region_uniformity(file_path_bloc2, inner_rectangle)
        print("Uniformly white?" if is_uniform else "Contains other elements")

        #bloc_2_info, bloc_2_sign = crop_image_around_reference(file_path_bloc2, list_coord_signa_bloc2)

        # If you want to save the cropped images
        #bloc_2_info_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_bloc_2_info.jpeg"))
        #bloc_2_sign_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_bloc_2_sign.jpeg"))

        #cv2.imwrite(bloc_2_info_path, bloc_2_info)
        #cv2.imwrite(bloc_2_sign_path, bloc_2_sign)
        #return bloc_2_info_path, bloc_2_sign_path


def find_custom_rectangle(rectangles):
    if not rectangles or len(rectangles) < 3:
        return None

    # Rectangle format: ((start_x, start_y), (end_x, end_y))
    top_rect = rectangles[0]     # Top rectangle
    bottom_rect = rectangles[1]  # Bottom rectangle
    right_rect = rectangles[2]   # Right limit rectangle

    # Bottom of the top rectangle (y coordinate)
    bottom_y_top_rect = top_rect[1][1]

    # Top of the bottom rectangle (y coordinate)
    top_y_bottom_rect = bottom_rect[0][1]

    # Left limit of the top rectangle (x coordinate)
    left_x_top_rect = top_rect[0][0]

    # Right limit of the right rectangle (x coordinate)
    right_x_right_rect = right_rect[0][0]

    return ((left_x_top_rect, bottom_y_top_rect), (right_x_right_rect, top_y_bottom_rect))

# Example usage
rectangles = [((3, 3), (690, 150)), ((6, 299), (721, 374)), ((569, 5), (675, 371))]
custom_rectangle = find_custom_rectangle(rectangles)
print(custom_rectangle)

import cv2
import numpy as np

def check_region_uniformity(image_path, rectangle):
    # Load the image
    image = cv2.imread(image_path)

    # Ensure the image was loaded successfully
    if image is None:
        print("Error: Image not found")
        return

    # Crop the region from the image
    # Rectangle format: ((start_x, start_y), (end_x, end_y))
    start_x, start_y = rectangle[0]
    end_x, end_y = rectangle[1]
    cropped_region = image[start_y:end_y, start_x:end_x]

    # Convert to grayscale for easier analysis
    gray_region = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)

    # Check if the region is uniformly white
    # You may need to adjust the threshold value depending on the definition of 'white' in your context
    white_threshold = 240 # Assuming pixel values above 240 (out of 255) are considered white
    if np.all(gray_region >= white_threshold):
        return True  # The region is uniformly white
    else:
        return False  # The region contains other elements





# +
output_fold = "data/performances_data/arval_classic_restitution_images/test_signa/"
file_path_bloc2 = "data/performances_data/arval_classic_restitution_images/test_signa/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_locataire client_p1_bloc_2_sign.jpeg"
file_name ="EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_locataire client_p1_bloc_2_sign.jpeg"
#file_path_bloc2 ="data/performances_data/arval_classic_restitution_images/test_signa/EQ-807-SZ_PV_de_reprise_EQ-807-SZ_p1_bloc_2_sign.jpeg"
template_path_signature_bloc2_top = 'data/performances_data/arval_classic_restitution_images/template/template_path_signature_bloc2_top.png'
template_path_signature_bloc2_bottom = 'data/performances_data/arval_classic_restitution_images/template/template_path_signature_bloc2_bottom.png'
template_path_signature_bloc2_right = 'data/performances_data/arval_classic_restitution_images/template/template_path_signature_bloc2_right.png'

from PIL import Image

image_path = template_path_signature_bloc2_bottom  
img = mpimg.imread(image_path)
plt.imshow(img)
plt.axis('off') 

print(get_image_dimensions(template_path_signature_bloc2_bottom))
print(get_image_dimensions(file_path_bloc2))

crop_bloc2_signature(file_path_bloc2, output_fold, file_name, template_path_signature_bloc2_top,template_path_signature_bloc2_bottom,template_path_signature_bloc2_right)

# Example usage
#rectangle = ((3, 150), (569, 299))
#is_uniform = check_region_uniformity(file_path_bloc2, rectangle)
#print("Uniformly white?" if is_uniform else "Contains other elements")


# +

# Set the directory where your images are stored
image_directory = Path('data/performances_data/arval_classic_restitution_images/')

template_path_top_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_le_vehicule.png'
template_path_top_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_descriptif.png'
template_path_bot_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_end_block3.png'
template_path_bot_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_barcode.png'

#template to subdivise the bloc:
template_path_signature_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_2_garage.png'
template_path_signature_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_4_long.png'

output_fold = "data/performances_data/arval_classic_restitution_images/test_signa/"

image_files = os.listdir(image_directory)
print(image_files)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        print('**************************************')
        print(f"Processing {file_path}")
        print('**************************************')

        try:
            # Getting bloc 2 and 4
            new_dimensions = get_image_dimensions(file_path)
            bloc2 = find_block2_infos(file_path, template_path_top_bloc2, template_path_top_bloc3, new_dimensions)
            bloc4 = find_block4_infos(file_path, template_path_bot_bloc3, template_path_bot_bloc4, new_dimensions)
            draw_contour_rectangles_on_image(file_path, [bloc2, bloc4])
            blocs = [bloc2, bloc4]
            print(blocs)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to get bloc 2 and 4 of ", file_name ," :", e)
            print('-----------------')
        
        
        image = cv2.imread(file_path)
        try:
            #cropping the image in bloc
            crop_blocks_in_image(image, blocs, output_fold, file_name)
            cropped_image_paths = [os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_{i}.jpeg") for i in range(len(blocs))]
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to crop the image ", file_name ," :", e)
            print('-----------------')

        #Dividing and cropping bloc 2:
        try:
            file_path_bloc2 = str(cropped_image_paths[0])
            bloc_2_info_path,bloc_2_sign_path = arval_classic_divide_and_crop_bloc2(file_path_bloc2,output_fold,file_name,template_path_signature_bloc2)

        
            image_path = bloc_2_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_2_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
            crop_bloc2_signature(bloc_2_sign_path, output_fold, file_name, template_path_signature_bloc2_top,template_path_signature_bloc2_bottom,template_path_signature_bloc2_right)
            
            out_croped_bloc2_path = output_fold +'bloc2_'+file_name
            print(out_croped_bloc2_path)
            print(bloc_2_sign_path)
            #place_image_in_center(bloc_2_sign_path, new_dimensions, out_croped_bloc2_path)

        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 2 in two", file_name ," :", e)
            print('-----------------')
        

        #Dividing and cropping bloc 4:
        try:
            file_path_bloc4 = str(cropped_image_paths[1])
            bloc_4_info_path,bloc_4_sign_path = arval_classic_divide_and_crop_bloc4(file_path_bloc4,output_fold,file_name,template_path_signature_bloc4)

        
            image_path = bloc_4_info_path  
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

            print('  ')

            image_path = bloc_4_sign_path  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 4 in two", file_name ," :", e)
            print('-----------------')
        

        
        #print(f"Processed {file_path}")
        print('-----------------------------------')
        print('   ')
        print('   ')
    else:
        print(f"Skipped non-image file: {file_name}")

    
    #break
    
print("All files have been processed.")
# -


