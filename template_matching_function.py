from PIL import Image,ImageDraw
import os
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import cv2
import numpy as np
import matplotlib.pyplot as plt

load_dotenv(find_dotenv())
PLOT_MATCHED_BLOCKS = os.environ.get('PLOT_MATCHED_BLOCKS')


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns a tuple (width, height)

# Function to perform template matching at multiple scales
def multi_scale_template_matching(image_path, template_path, scales, plot_img=False):

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

    img = cv2.imread(image_path.as_posix(), 0)
    logger.info(f"Image shape: {img.shape}")
    template = cv2.imread(template_path.as_posix(), 0)
    w, h = template.shape[::-1]
    logger.info(f"Template initial shape: {template.shape}")
    # List to store the results
    results = []

    # Initial scale range

    best_match_quality, best_match_coordinates, best_scale = get_best_match_to_template(img, template, scales, results)

    # Continue with smaller scales if needed
    #start_scale, end_scale = best_scale - 0.05, 0.2
    #smaller_scales = np.linspace(start_scale, end_scale, 20)
    #best_match_quality, best_match_coordinates, best_scale = get_best_match_to_template(img, template, smaller_scales, results)

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
        document_name_no_ext = Path(image_path).parent.name
        ax.set_title(f'Template matched:{template_name}\nScale: {best_scale} - Quality: {best_match_quality}')
        folder = Path(f"data/performances_data/invalid_data/arval_classic_restitution_images/tmp/{document_name_no_ext}/detected_templates")

        os.makedirs(folder, exist_ok=True)
        fig.savefig(folder / f"detected_{template_name}")
        plt.show()

    return output_image_path, best_match_quality, best_scale, best_match_coordinates


def remove_rectangle_from_image(image_path, output_path, top_left, bottom_right, plot_img=False):

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

    if plot_img:
        # Display the result
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Image with Rectangle Removed')
        plt.axis('off')
        plt.show()

    return output_image_path

def find_bottom_right_rectangle(rectangles):
    """
    Find the rectangle located the most to the bottom-right.
    Rectangles are given as a list of tuples, each containing top-left and bottom-right points (x, y).
    For example: [((x1, y1), (x2, y2)), ...]
    """
    if not rectangles:
        return None  # Return None if the list is empty

    # Initialize with the first rectangle
    bottom_right_rect = rectangles[0]

    # Iterate through the rectangles and find the one with the largest x and y in the bottom-right point
    for rect in rectangles:
        _, (br_x, br_y) = rect  # Unpack the bottom-right point
        _, (max_br_x, max_br_y) = bottom_right_rect  # Current bottom-right most rectangle

        # Update if this rectangle is more bottom-right
        if br_x > max_br_x or (br_x == max_br_x and br_y > max_br_y):
            bottom_right_rect = rect

    return bottom_right_rect

def find_top_rectangle(rectangles):
    """
    Find the rectangle located the most to the top.
    Rectangles are given as a list of tuples, each containing top-left and bottom-right points (x, y).
    For example: [((x1, y1), (x2, y2)), ...]
    """
    if not rectangles:
        return None  # Return None if the list is empty

    # Initialize with the first rectangle
    top_rect = rectangles[0]

    # Iterate through the rectangles and find the one with the smallest y (and x as secondary) in the top-left point
    for rect in rectangles:
        (tl_x, tl_y), _ = rect  # Unpack the top-left point
        (min_tl_x, min_tl_y), _ = top_rect  # Current top-most rectangle

        # Update if this rectangle is more top
        if tl_y < min_tl_y or (tl_y == min_tl_y and tl_x < min_tl_x):
            top_rect = rect

    return top_rect

def calculate_line_heights(top_rect, bottom_rect):
    """
    Calculate the heights of three horizontal lines based on the top and bottom rectangles.
    top_rect and bottom_rect are tuples with top-left and bottom-right points (x, y).
    Returns a tuple with three y-coordinates (y1, y2, y3).
    """
    _, (top_x2, top_y2) = top_rect  # Bottom of the top rectangle
    (bottom_x1, bottom_y1), _ = bottom_rect  # Top of the bottom rectangle

    mid_y = (top_y2 + bottom_y1) // 2  # Middle line between the two rectangles

    return top_y2, bottom_y1, mid_y


def resize_arval_classic(original_image_path, plot_image=False):
    original_image = Image.open(original_image_path)

    # Calculate the new width while maintaining the aspect ratio
    original_width, original_height = original_image.size
    new_height = 3508
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save the resized image
    #resized_image_path = 'resized_image.jpeg'
    #resized_image.save(resized_image_path)

    if plot_image:
        resized_image.show()
        print(new_width, new_height )
    return resized_image

def isolate_code_bar_part_arval_classic(image, output_image_save, plot_img=False):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw a vertical line in the middle of the image
    middle_x = width / 2
    draw.line([(middle_x, 0), (middle_x, height)], fill="black", width=2)

    # Draw five equally spaced horizontal lines across the image
    spacing = height / 6
    for i in range(1, 6):
        line_y = spacing * i
        draw.line([(0, line_y), (width, line_y)], fill="black", width=2)

    # Create a white rectangle over the entire image except for the bottom right section
    for i in range(6):  # There will be 5 horizontal sections
        for j in range(2):  # There will be 2 vertical sections
            if not (i == 5 and j == 1):  # Skip the bottom right section
                #print('i,j',i,j)
                top_left_corner = (j * middle_x, i * spacing)
                bottom_right_corner = ((j + 1) * middle_x, (i + 1) * spacing)
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")
                #image.show()

    if plot_img:
        image.show()
# Save the modified image
    image.save(output_image_save)

def get_code_bar_position(rezise_im, output_tmp_folder, template_path_bot_block4, plot_img):
    output_temp_file = output_tmp_folder / 'code_bar.jpeg'
    isolate_code_bar_part_arval_classic(rezise_im, output_temp_file, plot_img=plot_img)
    scales = np.linspace(0.26, 1.3, 20)
    output_image_path, best_match_quality, best_scale, best_match_coordinates = multi_scale_template_matching(output_temp_file, template_path_bot_block4, scales, plot_img=plot_img)
    return best_match_coordinates


def isolate_logo_part_arval_classic(image, output_image_path, plot_img=False):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw a vertical line in the middle of the image
    middle_x = width / 2
    draw.line([(middle_x, 0), (middle_x, height)], fill="black", width=2)

    # Draw five equally spaced horizontal lines across the image
    spacing = height / 6
    for i in range(1, 6):
        line_y = spacing * i
        draw.line([(0, line_y), (width, line_y)], fill="black", width=2)

    # Create a white rectangle over the entire image except for the bottom right section
    for i in range(6):  # There will be 5 horizontal sections
        for j in range(2):  # There will be 2 vertical sections
            if not (i == 0 and j == 0):  # Skip the bottom right section
                #print('i,j',i,j)
                top_left_corner = (j * middle_x, i * spacing)
                bottom_right_corner = ((j + 1) * middle_x, (i + 1) * spacing)
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")
                #image.show()

    if plot_img:
        image.show()
# Save the modified image
    image.save(output_image_path)

def get_top_bloc_position(resize_img, output_tmp_folder, template_path_top_block1, plot_img):
    output_temp_file = output_tmp_folder / 'arval_logo.jpeg'
    isolate_logo_part_arval_classic(resize_img, output_temp_file, plot_img=plot_img)
    scales = np.linspace(0.26, 1.3, 20)
    output_image_path, best_match_quality, best_scale, best_match_coordinates = multi_scale_template_matching(output_temp_file, template_path_top_block1, scales, plot_img=plot_img)
    return best_match_coordinates

def find_top_and_bot_of_arval_classic_restitution(rezise_img, output_tmp_folder, template_path_top_block1,
                                                  template_path_bot_block4, plot_img=False):
    copy_of_resize_img = rezise_img.copy()
    top_rect = get_top_bloc_position(copy_of_resize_img, output_tmp_folder, template_path_top_block1, plot_img=plot_img)
    copy_of_resize_img = rezise_img.copy()
    bottom_rect = get_code_bar_position(copy_of_resize_img, output_tmp_folder, template_path_bot_block4, plot_img=plot_img)

    if plot_img:
        draw = ImageDraw.Draw(rezise_img)

        # Draw rectangles
        for (top_left, bottom_right) in [top_rect, bottom_rect]:
            rect_coords = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]
            draw.rectangle(rect_coords, outline="red", width=3)

        # Calculate line heights
        top_line, bottom_line, mid_line = calculate_line_heights(top_rect, bottom_rect)

        # Draw horizontal lines
        img_width = rezise_img.width
        for y in [top_line, bottom_line, mid_line]:
            draw.line([(0, y), (img_width, y)], fill="green", width=2)
        rezise_img.show()
        #print(output_tmp_folder)
        #output_temp_file = output_tmp_folder + "/top_bot_temp_file.jpeg"
        #rezise_img.save(output_temp_file)

    return top_rect,bottom_rect

def multi_scale_multi_results_temp_matching(image_path, temp_path, output_path, number_of_iteration, plot_img=False, plot2=False):
    list_coord = []
    image_temp = image_path
    for i in range(number_of_iteration):
        try:
            output_image_path, match_quality, scale,best_match_coordinates =multi_scale_template_matching(image_temp, temp_path, plot_img)
        except Exception as e:
            print("No match found or an error occurred:", e)
        list_coord.append(best_match_coordinates)
        top_left, bottom_right = best_match_coordinates[0], best_match_coordinates[1]
        image_temp = remove_rectangle_from_image(image_temp, output_path, top_left, bottom_right, plot_img)
    if plot2:
        #draw_contour_rectangles_on_image(image_path, list_coord)

        # Display the result
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for (top_left, bottom_right) in list_coord:
            cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 3)  # Red color, 3 px thickness
        fig, ax = plt.subplots()
        ax.imshow(img)
        plt.show()

        template_name = Path(temp_path).name
        image_name = Path(image_path).name
        ax.set_title(f'Template matched:{template_name}\nScale:  - Quality: ')

        folder = output_path[:-5]
        fig.savefig(folder / f"{template_name[:-4][-8:]}_{image_name}")
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
        if len(cropped_image) == 0:
            logger.warning(f"Empty image detected. Skipping {output_filename}")
            continue
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



def isolate_top_bloc2(image, output_image_save, top_rect, bottom_rect, plot_img=False):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw a vertical line in the middle of the image
    middle_x = width / 2
    draw.line([(middle_x, 0), (middle_x, height)], fill="black", width=2)

    # Get the bottom of the top rectangle and the top of the bottom rectangle
    bottom_of_top_rect = top_rect[1][1]
    top_of_bottom_rect = bottom_rect[0][1]

    draw.line([(0, bottom_of_top_rect), (width, bottom_of_top_rect)], fill="black", width=2)
    draw.line([(0, top_of_bottom_rect), (width, top_of_bottom_rect)], fill="black", width=2)

    # Calculate the spacing for the three lines
    vertical_range = top_of_bottom_rect - bottom_of_top_rect
    spacing = vertical_range / 4

    # Draw three equally spaced horizontal lines
    for i in range(1, 4):
        line_y = bottom_of_top_rect + spacing * i
        draw.line([(0, line_y), (width, line_y)], fill="black", width=2)
    #image.show()

    # Create white rectangles over the entire image except for the specified section
    vertical_lines = [bottom_of_top_rect + i * spacing for i in range(1, 4)]
    vertical_lines = [bottom_of_top_rect] + vertical_lines + [top_of_bottom_rect]
    vertical_lines = [0.0] + vertical_lines + [height]

    for i in range(len(vertical_lines)-1):
        for j in range(2):  # There are 2 vertical sections, divided by middle_x
            # Define the corners of the rectangle
            top_left_corner = (j * middle_x, vertical_lines[i])
            bottom_right_corner = ((j+1) * middle_x, vertical_lines[i+1])

            # Skip the second block from the top on the left side
            if not (i == 1 and j == 0):
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")


    # Display the image if requested
    if plot_img:
        image.show()

    # Save the modified image
    image.save(output_image_save)

def get_top_bloc_2_position(rezise_im, output_temp_file, top_rect, bottom_rect, template_path_top_block2, plot_img=False):
    isolate_top_bloc2(rezise_im, output_temp_file, top_rect, bottom_rect, plot_img=plot_img)
    scales = np.linspace(0.26, 1.3, 20)
    output_image_path, best_match_quality, best_scale, best_match_coordinates = multi_scale_template_matching(output_temp_file, template_path_top_block2, scales, plot_img=plot_img)
    return best_match_coordinates


def isolate_top_bloc3(image, output_image_save, top_rect, bottom_rect, plot_img=False):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw a vertical line in the middle of the image
    middle_x = width*2 / 3
    draw.line([(middle_x, 0), (middle_x, height)], fill="black", width=2)

    # Get the bottom of the top rectangle and the top of the bottom rectangle
    bottom_of_top_rect = top_rect[1][1]
    top_of_bottom_rect = bottom_rect[0][1]

    draw.line([(0, bottom_of_top_rect), (width, bottom_of_top_rect)], fill="black", width=2)
    draw.line([(0, top_of_bottom_rect), (width, top_of_bottom_rect)], fill="black", width=2)

    # Calculate the spacing for the three lines
    vertical_range = top_of_bottom_rect - bottom_of_top_rect
    spacing = vertical_range / 4

    # Draw three equally spaced horizontal lines
    for i in range(1, 4):
        line_y = bottom_of_top_rect + spacing * i
        draw.line([(0, line_y), (width, line_y)], fill="black", width=2)
    #image.show()

    # Create white rectangles over the entire image except for the specified section
    vertical_lines = [bottom_of_top_rect + i * spacing for i in range(1, 4)]
    vertical_lines = [bottom_of_top_rect] + vertical_lines + [top_of_bottom_rect]
    vertical_lines = [0.0] +vertical_lines + [height]

    for i in range(len(vertical_lines)-1):
        for j in range(2):  # There are 2 vertical sections, divided by middle_x
            # Define the corners of the rectangle
            top_left_corner = (j * middle_x, vertical_lines[i])
            bottom_right_corner = ((j+1) * middle_x, vertical_lines[i+1])

            # Skip the second block from the top on the left side
            if not (i == 2 and j == 0):
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")


    # Display the image if requested
    if plot_img:
        image.show()

    # Save the modified image
    image.save(output_image_save)

def get_top_bloc_3_position(rezise_im, output_temp_file, top_rect, bottom_rect, template_path_top_block3, plot_img=False):
    isolate_top_bloc3(rezise_im, output_temp_file, top_rect, bottom_rect, plot_img=plot_img)
    scales = np.linspace(0.26, 1.3, 20)
    output_image_path, best_match_quality, best_scale, best_match_coordinates = multi_scale_template_matching(output_temp_file, template_path_top_block3, scales, plot_img=plot_img)
    return best_match_coordinates

def get_block2_rectangle(rezise_im, output_temp_file, top_rect, bottom_rect, template_path_top_block2, template_path_top_block3, plot_img=False):
    folder = Path(output_temp_file).parent
    width, height = rezise_im.size
    rezise_im1 = rezise_im.copy()
    output_temp_file = folder / 'top_block2.jpeg'

    top_bloc2 = get_top_bloc_2_position(rezise_im1, output_temp_file, top_rect, bottom_rect, template_path_top_block2,
                                        plot_img=plot_img)
    rezise_im2 = rezise_im.copy()
    output_temp_file = folder / 'top_block3.jpeg'
    top_bloc3 = get_top_bloc_3_position(rezise_im2, output_temp_file, top_rect, bottom_rect, template_path_top_block3,
                                        plot_img=plot_img)

    # Top of the first rectangle (y1 of top_rect)
    top_y = top_bloc2[0][1]

    # Top of the second rectangle (y1 of bottom_rect)
    bottom_y = top_bloc3[0][1]

    if plot_img:
        draw = ImageDraw.Draw(rezise_im)
        coord_rect = [(0, top_y), (width, bottom_y)]
        draw.rectangle(coord_rect, outline="red")

        # Get the image name and folder path
        image_name = Path(output_temp_file).name
        folder = Path(output_temp_file).parent

        # Construct the new file path for saving
        outputsave = folder / f"block2_rectangle.jpeg"

        # Save the image
        rezise_im.save(outputsave)
    return (0, top_y), (width, bottom_y)




def isolate_top_bloc4(image, output_image_save, top_rect, bottom_rect, plot_img=False):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Get image dimensions
    width, height = image.size

    # Draw a vertical line in the middle of the image
    middle_x = width / 2
    draw.line([(middle_x, 0), (middle_x, height)], fill="black", width=2)

    # Get the bottom of the top rectangle and the top of the bottom rectangle
    bottom_of_top_rect = top_rect[1][1]
    top_of_bottom_rect = bottom_rect[0][1]

    draw.line([(0, bottom_of_top_rect), (width, bottom_of_top_rect)], fill="black", width=2)
    draw.line([(0, top_of_bottom_rect), (width, top_of_bottom_rect)], fill="black", width=2)

    # Calculate the spacing for the three lines
    vertical_range = top_of_bottom_rect - bottom_of_top_rect
    spacing = vertical_range / 2

    # Draw three equally spaced horizontal lines
    for i in range(1, 2):
        line_y = bottom_of_top_rect + spacing * i
        draw.line([(0, line_y), (width, line_y)], fill="black", width=2)
    #image.show()

    # Create white rectangles over the entire image except for the specified section
    vertical_lines = [bottom_of_top_rect + i * spacing for i in range(1, 3)]
    vertical_lines = [bottom_of_top_rect] + vertical_lines + [top_of_bottom_rect]
    vertical_lines = [0.0] +vertical_lines + [height]

    for i in range(len(vertical_lines)-1):
        for j in range(2):  # There are 2 vertical sections, divided by middle_x
            # Define the corners of the rectangle
            top_left_corner = (j * middle_x, vertical_lines[i])
            bottom_right_corner = ((j+1) * middle_x, vertical_lines[i+1])

            # Skip the second block from the top on the left side
            if not (i == 2 and j == 0):
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")


    # Display the image if requested
    if plot_img:
        image.show()

    # Save the modified image
    image.save(output_image_save)

def get_top_bloc_4_position(rezise_im, output_temp_file, top_rect, bottom_rect, template_path_top_block4, plot_img=False):
    isolate_top_bloc4(rezise_im, output_temp_file, top_rect, bottom_rect, plot_img=plot_img)
    scales = np.linspace(0.26, 1.3, 20)
    output_image_path, best_match_quality, best_scale, best_match_coordinates = multi_scale_template_matching(output_temp_file, template_path_top_block4, scales, plot_img=plot_img)
    return best_match_coordinates

def get_block4_rectangle(rezise_im, output_temp_file, top_rect, bottom_rect, template_path_top_block4, plot_img=False):
    folder = Path(output_temp_file).parent

    width, height = rezise_im.size
    rezise_im1 = rezise_im.copy()
    output_temp_file = folder / 'bottom_left.jpeg'

    top_bloc4 = get_top_bloc_4_position(rezise_im1, output_temp_file, top_rect, bottom_rect, template_path_top_block4,
                                        plot_img=plot_img)

    bot_bloc4 = bottom_rect

    # Top of the first rectangle (y1 of top_rect)
    top_y = top_bloc4[1][1]

    # Top of the second rectangle (y1 of bottom_rect)
    bottom_y = bot_bloc4[0][1]

    if plot_img:
        draw = ImageDraw.Draw(rezise_im)
        coord_rect = [(0, top_y), (width, bottom_y)]
        draw.rectangle(coord_rect, outline="red")

        # Get the image name and folder path
        image_name = Path(output_temp_file).name

        # Construct the new file path for saving
        outputsave = folder / f"block4_{image_name}"

        # Save the image
        rezise_im.save(outputsave)
    return (0, top_y), (width, bottom_y)

def draw_rectangles_and_save(image, rectangles, output_path):
    """
    Draw rectangles and save file
    """
    # Load the image
    draw = ImageDraw.Draw(image)

        # Draw each rectangle
    for rect in rectangles:
        # rect is in the format ((x1, y1), (x2, y2))
        draw.rectangle(rect, outline="red")

    # Save the modified image
    image.save(output_path)


def divide_and_fill_image3_5(image_path, output_image_save, plot_img=False):
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Calculate the width of the two blocks (3/5 and 2/5)
    left_block_width = int(width * 3 / 5)
    right_block_width = width - left_block_width

    # Draw a vertical line to separate the two blocks
    draw.line([(left_block_width, 0), (left_block_width, height)], fill="black", width=2)

    # Fill the left block (3/5 of the image) with white
    draw.rectangle([(0, 0), (left_block_width, height)], fill="white")

    # Display the image if requested
    if plot_img:
        image.show()

    # Save the modified image
    image.save(output_image_path)

#Dividing bloc 2:
def arval_classic_divide_and_crop_block2(file_path_block2, output_fold, file_name, template_path_signature_block2) :

        output_tmp_image_save = output_fold / "tmp_dividing_block.jpg"
        divide_and_fill_image3_5(file_path_block2, output_tmp_image_save, plot_img=False)
        scales = np.linspace(0.26, 1.3, 20)
        output_image_path, match_quality, scale, list_coord_signa_block2 = multi_scale_template_matching(output_tmp_image_save,
                                                                                                         template_path_signature_block2,scales,
                                                                                                         plot_img=PLOT_MATCHED_BLOCKS)

        block_2_info, block_2_sign = crop_image_around_reference(file_path_block2, list_coord_signa_block2)
        
        # If you want to save the cropped images
        block_2_info_path = output_fold / f"{os.path.splitext(file_name)[0]}_block_2_info.jpeg"
        block_2_sign_path = output_fold / f"{os.path.splitext(file_name)[0]}_block_2_sign.jpeg"
        
        cv2.imwrite(block_2_info_path, block_2_info)
        cv2.imwrite(block_2_sign_path, block_2_sign)
        return block_2_info_path, block_2_sign_path


def divide_and_fill_image_1_3(image_path, output_image_save, plot_img=False):
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)

    # Calculate the dimensions of each block
    block_width = width // 3
    block_height = height // 2

    # Fill each block with white, except the top middle one
    for i in range(3):  # Three blocks in width
        for j in range(2):  # Two blocks in height
            if not (i == 1 and j == 0):  # Skip the top middle block
                top_left_corner = (i * block_width, j * block_height)
                bottom_right_corner = ((i + 1) * block_width, (j + 1) * block_height)
                draw.rectangle([top_left_corner, bottom_right_corner], fill="white")

    # Display the image if requested
    if plot_img:
        image.show()

    # Save the modified image
    image.save(output_image_path)

def arval_classic_divide_and_crop_block4(file_path_block4,output_fold,file_name,template_path_signature_block4) :
        output_tmp_image_save = output_fold / "tmp_dividing_block.jpg"
        divide_and_fill_image_1_3(file_path_block4, output_tmp_image_save, plot_img=False)
        scales = np.linspace(0.26, 1.3, 20)
        output_image_path, match_quality, scale, list_coord_signa_block4 = multi_scale_template_matching(output_tmp_image_save,
                                                                                                         template_path_signature_block4, scales,plot_img=PLOT_MATCHED_BLOCKS)

        block_4_info, block_4_sign = crop_image_around_reference2(file_path_block4, list_coord_signa_block4)
        
        # If you want to save the cropped images
        block_4_info_path = output_fold / f"{os.path.splitext(file_name)[0]}_{'block_4_info'}.jpeg"
        block_4_sign_path = output_fold / f"{os.path.splitext(file_name)[0]}_{'block_4_sign'}.jpeg"
        
        cv2.imwrite(block_4_info_path, block_4_info)
        cv2.imwrite(block_4_sign_path, block_4_sign)
    
        return block_4_info_path, block_4_sign_path