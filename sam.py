from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import supervision as sv
#import torch
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
import os
from loguru import logger
import cv2
from PIL import Image


load_dotenv(find_dotenv())
DEVICE = os.environ.get('DEVICE')

#SAM pre template-matching_processing function:

model_type = "vit_b"

HOME = os.getcwd()
#CHECKPOINT_PATH = os.path.join(HOME, "sam_weights", "sam_vit_h_4b8939.pth")
CHECKPOINT_PATH = os.path.join(HOME, "sam_weights", "sam_vit_b_01ec64.pth")

#Check if the weight exist, if they don't we download them
#print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

if os.path.isfile(CHECKPOINT_PATH):
    logger.info(f"SAM weights located in {CHECKPOINT_PATH}")
else:
    logger.info("Downloading SAM weights")
    os.makedirs(os.path.join(HOME, 'sam_weights'), exist_ok=True)
    os.system(f'wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P {os.path.join(HOME, "sam_weights")}')
    logger.info(f"{CHECKPOINT_PATH}; exists: {os.path.isfile(CHECKPOINT_PATH)}")


sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)

#If cuda available uncomment that line
sam.to(device=DEVICE)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=4,
    pred_iou_thresh=0.82,
    stability_score_thresh=0.9,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    #min_mask_region_area=400,  # Requires open-cv to run post-processing
)


def detect_page(img_path, sam_model=mask_generator_2, plot_option=False):
    """
    Detect
    :param img_path:
    If the page is already properly scanned, then it returns it, otherwise it crops it and returns the cropped image.
    :param sam_model:
    :param plot_option:
    :return:
    """
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get dimensions of the image
    height, width, channels = image_rgb.shape

    logger.info(f"Width and height of the image before SAM:{width}, {height}")

    sam_result = sam_model.generate(image_rgb)
    logger.info(3)
    selected_masks = []

    min_area = width * height * 60 / 100
    max_area = width * height

    for mask in sam_result:

        if mask['area'] > min_area and mask['area'] < max_area:
            selected_masks.append(mask)

    if len(selected_masks) < 1:
        if plot_option:
            plt.imshow(image_rgb)
        return image_rgb
    else:
        logger.info('At least one mask was found by SAM, cropping...')
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        detections = sv.Detections.from_sam(sam_result=selected_masks)

        annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
        bbox = selected_masks[0]['bbox'] #TODO: make sure that we pick the biggest one, not the first one

        cropped_image = image_bgr[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2])]
        height_c, width_c, channels = cropped_image.shape
        logger.info('width/height ratio after SAM cropping : {width_c}/ {height_c} ')
        if plot_option:
            sv.plot_images_grid(
                images=[image_bgr, annotated_image, cropped_image],
                grid_size=(1, 3),
                titles=['source image', 'segmented image', 'cropped_image']
            )
        return cropped_image

def rotate_image_if_needed(image_path, output_path):
    # Load the image
    original_image = Image.open(image_path)

    # Calculate the width/height ratio
    ratio = original_image.width / original_image.height

    # Rotate the image if the ratio is not between 0.65 and 0.85
    if not (ratio <= 0.85):
        rotated_image = original_image.rotate(90, expand=True)
        rotated_image.save(output_path)
        return "Image rotated and saved to " + str(output_path)
    else:
        original_image.save(output_path)
        return "No rotation needed."

def is_upside_down(image_path):
    # Load the image
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    # Use pytesseract to get orientation information
    ocr_data = pytesseract.image_to_osd(image, output_type=Output.DICT)
    return ocr_data['rotate'] == 180

def rotate_image(image_path, degrees, output_path):
    # Load the image
    image = Image.open(image_path)

    # Rotate the image
    rotated_image = image.rotate(degrees, expand=True)
    rotated_image.save(output_path)
    return output_path


def sam_pre_template_matching_function(img_path, output_folder, plot_option=False):
    """
    This function is used to crop the image with SAM, rotate it if needed and flip it if needed.
    :param img_path:
    :param output_path:
    :param plot_option:
    :return:
    """
    # Cropping the image with SAM
    image = detect_page(img_path, plot_option=plot_option)
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder / 'cropped_by_SAM.jpeg'
    cv2.imwrite(output_path.as_posix(), image)

    # Rotating
    rotate_image_if_needed(output_path, output_path)

    # Test if the rotation was correct or flip it by 180°
    if is_upside_down(output_path):
        # Rotate the image 180 degrees
        output_path = rotate_image(output_path, 180, output_path)
        logger.info(f"Image was upside down and has been rotated. Saved to {output_path}")
    else:
        logger.info("Image is not upside down.")

    return output_path

