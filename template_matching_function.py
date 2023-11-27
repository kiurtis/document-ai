from PIL import Image
import os
from loguru import logger
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

load_dotenv(find_dotenv())
PLOT_MATCHED_BLOCKS = os.environ.get('PLOT_MATCHED_BLOCKS')

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns a tuple (width, height)

# Function to perform template matching at multiple scales
def multi_scale_template_matching(image_path, template_path, plot_img=False):

    def apply_template_matching(img, template, scale, results):
        resized_template = cv2.resize(template, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        w_resized, h_resized = resized_template.shape[::-1]

        res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        results.append((max_val, (max_loc, (max_loc[0] + w_resized, max_loc[1] + h_resized)), scale))

        return results

    def get_best_match_to_template(img, template, scales, results):
        # Perform initial template matching
        for scale in scales:
            try:
                results = apply_template_matching(img, template, scale, results)
            except cv2.error as e:
                logger.warning(f"Error {e} while applying template matching at scale {scale}")
        # Sort the results by match quality
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        best_match_quality, best_match_coordinates, best_scale = sorted_results[0]
        return best_match_quality, best_match_coordinates, best_scale

    img = cv2.imread(image_path, 0)
    logger.info(f"Image shape: {img.shape}")
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    logger.info(f"Template initial shape: {template.shape}")
    # List to store the results
    results = []

    # Initial scale range
    scales = np.linspace(0.3, 1.5, 20)
    best_match_quality, best_match_coordinates, best_scale = get_best_match_to_template(img, template, scales, results)

    # Continue with smaller scales if needed
    start_scale, end_scale = best_scale - 0.05, 0.2
    smaller_scales = np.linspace(start_scale, end_scale, 20)
    best_match_quality, best_match_coordinates, best_scale = get_best_match_to_template(img, template, smaller_scales, results)

    # Draw a rectangle around the matched region
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(img_rgb, best_match_coordinates[0], best_match_coordinates[1], (0, 255, 0), 2)

    # Save the result image
    output_image_path = 'match_result_updated.png'
    cv2.imwrite(output_image_path, img_rgb)
    if plot_img:
        fig, ax = plt.subplots(figsize=(16, 12))
        # Display the result
        ax.imshow(img_rgb)
        template_name = Path(template_path).name
        image_name = Path(image_path).name
        ax.set_title(f'Template matched:{template_name}\nScale: {best_scale} - Quality: {best_match_quality}')
        folder = "/Users/amielsitruk/work/terra_cognita/customers/pop_valet/ai_documents/data/performances_data/invalid_data/arval_classic_restitution_images/tmp/detected_templates/"
        fig.savefig(folder + template_name + "_" + image_name)
        plt.show()

    return output_image_path, best_match_quality, best_scale, best_match_coordinates


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



def multi_scale_multi_results_temp_matching(image_path, temp_path,output_path, number_of_iteration,print_img=False, print2=False):
    list_coord = []
    image_temp = image_path
    for i in range(number_of_iteration):
        try:
            output_image_path, match_quality, scale,best_match_coordinates =multi_scale_template_matching(image_temp, temp_path, print_img)
        except Exception as e:
            print("No match found or an error occurred:", e)
        list_coord.append(best_match_coordinates)
        top_left, bottom_right = best_match_coordinates[0], best_match_coordinates[1]
        image_temp = remove_rectangle_from_image(image_temp, output_path ,top_left, bottom_right,print_img)
    if print2:
        draw_contour_rectangles_on_image(image_path, list_coord)
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
    coord = int(top_rect[1][1]*99/100)
    block4 = ((0, coord), (page_width, bottom_rect[0][1]))
    
    return block4

def draw_contour_rectangles_on_image(image_path, rectangles):
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



# Function to crop and save images
def crop_blocks_in_image(image, blocks, output_dir, input_filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_filename, file_extension = os.path.splitext(input_filename)
    for i, block in enumerate(blocks):
        (x1, y1), (x2, y2) = block
        cropped_image = image[y1:y2, x1:x2]
        output_filename = f"{base_filename}_{i}.jpeg"
        logger.info(f"Saving cropped image to {output_dir}/{output_filename}")
        cv2.imwrite(os.path.join(output_dir, output_filename), cropped_image)


def crop_image_around_reference(image_path, reference_rect):
    # Read the image
    image = cv2.imread(image_path)

    # Unpack the reference rectangle coordinates
    (ref_x1, ref_y1), (_, _) = reference_rect

    # Determine the full height of the image
    full_height = image.shape[0]

    # Crop the image into two parts
    left_crop = image[0:full_height, 0:ref_x1]
    right_crop = image[0:full_height, ref_x1:image.shape[1]]

    return left_crop, right_crop


def crop_image_around_reference2(image_path, reference_rect):
    # Read the image
    image = cv2.imread(image_path)

    # Unpack the reference rectangle coordinates
    (ref_x1, ref_y1), (ref_x2, ref_y2) = reference_rect

    # Determine the full height of the image
    full_height = image.shape[0]

    # Crop the image into two parts
    # The left crop goes from the start to the end of the reference rectangle
    # The right crop goes from the end of the reference rectangle to the end of the image
    left_crop = image[0:full_height, 0:ref_x2]
    right_crop = image[0:full_height, ref_x2:image.shape[1]]

    return left_crop, right_crop




def find_block2_infos(file_path, template_path_top_block2, template_path_top_block3, file_dimensions) :
    #top block2 coordinate
    logger.info(f"Getting block 2 bottom following template: {template_path_top_block2.split('/')[-1]}")
    output_image_path, match_quality, scale, list_coord_top_block2 = multi_scale_template_matching(file_path,
                                                                                                   template_path_top_block2,
                                                                                                   plot_img=PLOT_MATCHED_BLOCKS)

    #bottom block2 coordinate
    logger.info(f"Getting block 3 top following template: {template_path_top_block3.split('/')[-1]}...")
    output_image_path, match_quality, scale, list_coord_top_block3 = multi_scale_template_matching(file_path,
                                                                                                   template_path_top_block3,
                                                                                                   plot_img=PLOT_MATCHED_BLOCKS)
    
    #Creating bloc 2
    block2 = create_block2_from_ref_rectangles(list_coord_top_block2, list_coord_top_block3, file_dimensions[1])

    return block2


def find_block4_infos(file_path, template_path_bot_block3, template_path_bot_block4, file_dimensions) :
    #bottom block3 coordinate
    logger.info(f"Getting block 3 bottom following templates {template_path_bot_block3.split('/')[-1]}")
    output_image_path, match_quality, scale, list_coord_bot_block3 = multi_scale_template_matching(file_path, template_path_bot_block3, plot_img=PLOT_MATCHED_BLOCKS)
    
    #bottom block4 coordinate
    logger.info(f"Getting block 4 bottom following templates {template_path_bot_block4.split('/')[-1]}")
    output_image_path, match_quality, scale, list_coord_bot_block4 = multi_scale_template_matching(file_path, template_path_bot_block4, plot_img=PLOT_MATCHED_BLOCKS)
    
    #Creating bloc 4
    block4 = create_block4_from_ref_rectangles(list_coord_bot_block3, list_coord_bot_block4, file_dimensions[1])

    return block4

#block4 = arval_classic_create_block4(file, template_path_bot_block3,template_path_top_block3,file_dimensions) 



#Dividing bloc 2:
def arval_classic_divide_and_crop_block2(file_path_block2, output_fold, file_name, template_path_signature_block2) :
        
        output_image_path, match_quality, scale, list_coord_signa_block2 = multi_scale_template_matching(file_path_block2,
                                                                                                         template_path_signature_block2,
                                                                                                         plot_img=PLOT_MATCHED_BLOCKS)

        block_2_info, block_2_sign = crop_image_around_reference(file_path_block2, list_coord_signa_block2)
        
        # If you want to save the cropped images
        block_2_info_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_block_2_info.jpeg"))
        block_2_sign_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_block_2_sign.jpeg"))
        
        cv2.imwrite(block_2_info_path, block_2_info)
        cv2.imwrite(block_2_sign_path, block_2_sign)
        return block_2_info_path, block_2_sign_path



def arval_classic_divide_and_crop_block4(file_path_block4,output_fold,file_name,template_path_signature_block4) :
        
        output_image_path, match_quality, scale, list_coord_signa_block4 = multi_scale_template_matching(file_path_block4,
                                                                                                         template_path_signature_block4, plot_img=PLOT_MATCHED_BLOCKS)

        block_4_info, block_4_sign = crop_image_around_reference2(file_path_block4, list_coord_signa_block4)
        
        # If you want to save the cropped images
        block_4_info_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_{'block_4_info'}.jpeg"))
        block_4_sign_path = str(os.path.join(output_fold, f"{os.path.splitext(file_name)[0]}_{'block_4_sign'}.jpeg"))
        
        cv2.imwrite(block_4_info_path, block_4_info)
        cv2.imwrite(block_4_sign_path, block_4_sign)
    
        return block_4_info_path,block_4_sign_path


