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
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.image as mpimg

from template_matching_function import get_image_dimensions,arval_classic_create_bloc2,arval_classic_create_bloc4,contour_rectangles_on_image,crop_and_save_image,arval_classic_divide_and_crop_bloc2,arval_classic_divide_and_crop_bloc4


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

output_fold = 'data/performances_data/arval_classic_restitution_images/output_cropped_images/'

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
            bloc2 = arval_classic_create_bloc2(file_path, template_path_top_bloc2, template_path_top_bloc3, new_dimensions)
            bloc4 = arval_classic_create_bloc4(file_path, template_path_bot_bloc3, template_path_bot_bloc4, new_dimensions)
            contour_rectangles_on_image(file_path, [bloc2, bloc4])
            blocs = [bloc2, bloc4]
            print(blocs)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to get bloc 2 and 4 of ", file_name ," :", e)
            print('-----------------')
        
        
        image = cv2.imread(file_path)
        try:
            #cropping the image in bloc
            crop_and_save_image(image, blocs, output_fold, file_name)
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

    
    #break
    
print("All files have been processed.")
# -


