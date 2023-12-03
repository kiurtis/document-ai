import os 
import cv2
import random
from bisect import bisect_left
from pipeline import get_processed_boxes_and_words_unguided_bloc
from utils import clean_listdir

#Functions that find non-crossing lines and cut the doc:

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




def subdivising_image(input_img,out_bloc_folder,line_number,hyperparameters,verbose):
    converted_boxes,image_dims = get_processed_boxes_and_words_unguided_bloc(img_path=input_img,
                                                            det_arch=hyperparameters['det_arch'],
                                                            reco_arch=hyperparameters['reco_arch'],
                                                            pretrained=hyperparameters['pretrained'],
                                                            verbose=verbose)
    img_height = image_dims[1]
    non_crossing_lines = find_max_spacing_non_crossing_lines(converted_boxes, img_height,line_number)
    sorted_lines = sorted(non_crossing_lines)
    cut_and_save_image(input_img, out_bloc_folder, sorted_lines)



def subdivide_batch_of_image(path_to_folder,line_number,hyperparameters,verbose):

    print(path_to_folder)
    image_list = clean_listdir(path_to_folder,only = "dir")
    print(image_list)

    for element in image_list:
        print(f'==== Running for file: {element} =====')
        filename_prefix = f'{element[:-5]}'

        local_fold = path_to_folder/element
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
        subdivising_image(path_to_name_jpeg,subfolder_path,line_number,hyperparameters,verbose)
