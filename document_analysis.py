import PIL
import json
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger
from pathlib import Path



#Importing functions
from utils import get_result_template, clean_listdir
from template_matching_function import get_image_dimensions,sam_pre_template_matching_function,\
    draw_contour_rectangles_on_image,crop_blocks_in_image, arval_classic_divide_and_crop_block2, arval_classic_divide_and_crop_block4,\
    find_top_and_bot_of_arval_classic_restitution,resize_arval_classic,get_bloc2_rectangle,get_bloc4_rectangle,draw_rectangles_and_save


from pipeline import get_processed_boxes_and_words,postprocess_boxes_and_words_arval_classic_restitution
from document_parsing import find_next_right_word
from image_processing import get_image_orientation, rotate_image
from gpt import build_block_checking_payload, request_completion, build_overall_quality_checking_payload, build_signature_checking_payload
from plotting import plot_boxes_with_text
from performance_estimation import has_found_box


class ArvalClassicDocumentAnalyzer:
    def __init__(self, document_name, path_to_document, hyperparameters):
        self.document_name = document_name
        self.path_to_document = path_to_document
        self.results = {}  # To be filled with results of analysis
        self.folder_path = os.path.dirname(self.path_to_document)  # Folder where the file is
        self.tmp_folder_path = os.path.join(self.folder_path, "tmp")  # Folder where we'll store the blocks
        self.hyperparameters = hyperparameters

        # Templates used to process the template matching
        self.template_path_top_block1 = 'data/performances_data/template/arval_classic_restitution/template_top_left.png'
        self.template_path_bot_block4 = 'data/performances_data/template/arval_classic_restitution/template_barcode.png'
        self.template_path_top_block2 = 'data/performances_data/template/arval_classic_restitution/template_path_top_block2.png'
        self.template_path_top_block3 = 'data/performances_data/template/arval_classic_restitution/template_descriptif.png'
        self.template_path_top_block4 = 'data/performances_data/template/arval_classic_restitution/template_end_block3.png'

        # Templates to subdivise the bloc:
        self.template_path_signature_block2 = 'data/performances_data/template/arval_classic_restitution/template_block_2_garage.png'
        self.template_path_signature_block4 = 'data/performances_data/template/arval_classic_restitution/template_block_4_long.png'

        if not os.path.exists(self.tmp_folder_path):
            os.makedirs(self.tmp_folder_path)

    def test_block_2_subdivision_existence(self):
        """
        For each block, test if it exists. If one is missing, it return False.
        """
        block_doc = []
        file_name = ['block_2_info', 'block_2_sign']
        missing_files = []

        for i in file_name:
            cropped_image_path = os.path.join(self.tmp_folder_path,
                                              f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")

            if os.path.exists(cropped_image_path):
                block_doc.append(cropped_image_path)
            else:
                missing_files.append(cropped_image_path)

        if len(missing_files) == 0:

            return True
        else:
            return False

    def test_block_4_subdivision_existence(self):
        """
        For each block, test if it exists. If one is missing, it return False.
        """
        block_doc = []
        file_name = ['block_4_info', 'block_4_sign']
        missing_files = []

        for i in file_name:
            cropped_image_path = os.path.join(self.tmp_folder_path,
                                              f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")

            if os.path.exists(cropped_image_path):
                block_doc.append(cropped_image_path)
            else:
                missing_files.append(cropped_image_path)

        if len(missing_files) == 0:

            return True
        else:
            return False

    def test_block_existence(self):
        """
        For each block, test if it exists. If one is missing, it return False.
        """
        block_doc = []
        file_name = ['block_2_info', 'block_2_sign', 'block_4_info', 'block_4_sign']
        missing_files = []

        for i in file_name:
            cropped_image_path = os.path.join(self.tmp_folder_path,
                                              f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")

            if os.path.exists(cropped_image_path):
                block_doc.append(cropped_image_path)
            else:
                missing_files.append(cropped_image_path)

        if len(missing_files) == 0:

            return True
        else:
            return False

    def test_block_2_and_4_existence(self):
        """
        For block 2 and 4, test if it exists. If one is missing, it return False.
        """
        block_doc = []
        missing_files = []

        for i in range(2):
            cropped_image_path = os.path.join(self.tmp_folder_path,
                                              f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")

            if os.path.exists(cropped_image_path):
                block_doc.append(cropped_image_path)
            else:
                missing_files.append(cropped_image_path)

        if len(missing_files) == 0:
            return True
        else:
            return False

    def read_block_2_and_4_path(self):
        """
        Create the block 2 and 4 path attributes
        """
        cropped_image_paths = []
        for i in range(2):
            cropped_image_paths.append(os.path.join(self.tmp_folder_path,
                                                    f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg"))

        self.file_path_block2 = str(cropped_image_paths[0])
        self.file_path_block4 = str(cropped_image_paths[1])


    def read_block2_subdivised_path(self):
        """
        Create the block 2 subdivision path attributes
        """
        file_name = ['block_2_info', 'block_2_sign']

        for file_name in file_name:
            path = os.path.join(self.tmp_folder_path, f"{os.path.splitext(self.document_name)[0]}_{file_name}.jpeg")
            setattr(self, f"{file_name}_path", path)

    def read_block4_subdivised_path(self):
        """
        Create the block 4 subddivision path attributes
        """
        file_name = ['block_4_info', 'block_4_sign']

        for file_name in file_name:
            path = os.path.join(self.tmp_folder_path, f"{os.path.splitext(self.document_name)[0]}_{file_name}.jpeg")
            setattr(self, f"{file_name}_path", path)


    def cropping_block2_and_4(self):
        """
        Divide the arval_classic_restitution type document in 4 parts and save them in self.tmp_folder_path.
        """

        try:
            # Temporary file:
            logger.info("Using SAM to crop image...")
            output_temp_file_sam = str(self.tmp_folder_path) + '/SAM_' + self.document_name
            sam_pre_template_matching_function(str(self.path_to_document), output_temp_file_sam, plot_option=True)

        except Exception as e:
            logger.error(f"An error occurred trying to use SAM {self.document_name}:{e}")


        try:
            # Getting block 2 and 4
            # Temporary file:
            output_temp_file = str(self.tmp_folder_path) + '/temps_' + self.document_name

            if os.path.exists(output_temp_file_sam):
                resize_im = resize_arval_classic(output_temp_file_sam)
            else:
                resize_im = resize_arval_classic(str(self.path_to_document))

            # Resizing image:
            copy_of_rezise_im = resize_im.copy()

            # Finding the bottom and the top of the document :
            top_rect, bottom_rect = find_top_and_bot_of_arval_classic_restitution(copy_of_rezise_im, output_temp_file,
                                                                                  self.template_path_top_block1,
                                                                                  self.template_path_bot_block4,
                                                                                  plot_img=False)
            copy_of_rezise_im = resize_im.copy()

            # Searching block2
            logger.info("Getting blocks 2...")
            block2 = get_bloc2_rectangle(copy_of_rezise_im, output_temp_file, top_rect, bottom_rect,
                                         self.template_path_top_block2, self.template_path_top_block3, plot_img=True)
            logger.info("Getting blocks 4...")
            copy_of_rezise_im = resize_im.copy()
            block4 = get_bloc4_rectangle(copy_of_rezise_im, output_temp_file, block2, bottom_rect,
                                         self.template_path_top_block4, plot_img=True)

            copy_of_rezise_im = resize_im.copy()
            draw_rectangles_and_save(copy_of_rezise_im, [block2, block4], output_temp_file)

            # draw_contour_rectangles_on_image(str(self.path_to_document), [block2, block4])
            blocks = [block2, block4]

        except Exception as e:
            logger.error(f"An error occurred trying to get blocks 2 and 4 of {self.document_name}:{e}")

        try:
            # Cropping and saving the blocks images in the tmp folder
            image = np.array(resize_im)
            logger.info("Cropping blocks...")

            crop_blocks_in_image(image, blocks,
                                 self.tmp_folder_path,
                                 self.document_name)
            cropped_image_paths = [os.path.join(self.tmp_folder_path,
                                                f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")
                                   for i in range(len(blocks))]
            self.file_path_block2 = str(cropped_image_paths[0])
            self.file_path_block4 = str(cropped_image_paths[1])

        except Exception as e:
            logger.error(f"An error occurred trying to crop the image {self.document_name}:{e}")

    def subdivising_and_cropping_block2(self):
        # Dividing and cropping block 2 in sub-blocks:
        try:
            logger.info("Dividing block 2...")
            self.block_2_info_path, self.block_2_sign_path = arval_classic_divide_and_crop_block2(self.file_path_block2,
                                                                                                  self.tmp_folder_path,
                                                                                                  self.document_name,
                                                                                                  self.template_path_signature_block2
                                                                                                  )
        except Exception as e:

            logger.error(f"An error occurred trying to divide block 2 in two {self.document_name}:{e}")

    def subdivising_and_cropping_block4(self):
        # Dividing and cropping block 4 in sub-blocks:
        try:
            logger.info("Dividing block 4...")
            self.block_4_info_path, self.block_4_sign_path = arval_classic_divide_and_crop_block4(self.file_path_block4,
                                                                                                  self.tmp_folder_path,
                                                                                                  self.document_name,
                                                                                                  self.template_path_signature_block4
                                                                                                  )
        except Exception as e:
            logger.error(f"An error occurred trying to divide block 4 in two {self.document_name}:{e}")

    def get_or_create_blocks(self):
        """
        Get the blocks: Create them if they don't exist or just retrieve them if they're already in self.tmp_folder_path
        """
        logger.info(f'Getting blocks...')
        if self.test_block_2_and_4_existence():
            logger.info(f'Blocks 2 and 4 already in tmp folder')
            self.read_block_2_and_4_path()
            if self.test_block_2_subdivision_existence():
                logger.info(f'Blocks 2 subdivisions already in tmp folder')
                self.read_block2_subdivised_path()
            else :
                self.subdivising_and_cropping_block2()

            if self.test_block_4_subdivision_existence():
                logger.info(f'Blocks 4 subdivisions already in tmp folder')
                self.read_block4_subdivised_path()
            else :
                self.subdivising_and_cropping_block4()

        else:
            self.cropping_block2_and_4()
            self.subdivising_and_cropping_block2()
            self.subdivising_and_cropping_block4()

    def plot_blocks(self):
        for image_path in [self.block_2_info_path, self.block_2_sign_path,
                           self.block_4_info_path, self.block_4_sign_path]:
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

    def get_result_template(self):
        """
        Get the template of the result json file, divided by blocks and including all the keywords.
        :return:
        """
        folder_ground_truths = Path('data/performances_data/valid_data/arval_classic_restitution_json/')
        self.template = get_result_template(folder_ground_truths)

    def analyze_block2_text(self,block2_text_image_path, verbose=False, plot_boxes=False):

        self.result_json_block_2 = {}

        converted_boxes = get_processed_boxes_and_words(img_path=block2_text_image_path,
                                                        block='block_2',
                                                        det_arch=self.hyperparameters['det_arch'],
                                                        reco_arch=self.hyperparameters['reco_arch'],
                                                        pretrained=self.hyperparameters['pretrained'],
                                                        verbose=verbose)

        converted_boxes = postprocess_boxes_and_words_arval_classic_restitution(converted_boxes,
                                                                                block='block_2',
                                                                                verbose=verbose,
                                                                                safe=True)
        if plot_boxes:
            file_dimensions = get_image_dimensions(block2_text_image_path)
            plot_boxes_with_text(converted_boxes, file_dimensions)

        for key_word in self.template['block_2']:
            if verbose:
                logger.info(f'Running {key_word}')
            self.result_json_block_2[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                      distance_margin=self.hyperparameters
                                                                          ['distance_margin'],
                                                                      max_distance=self.hyperparameters['max_distance'],
                                                                      minimum_overlap=self.hyperparameters
                                                                          ['minimum_overlap'],
                                                                      verbose=verbose)

            if isinstance(self.result_json_block_2[key_word], dict):
                self.result_json_block_2[key_word] = self.result_json_block_2[key_word]['next'][1]



    def analyze_block4_text(self ,block4_text_image_path,verbose=False ,plot_boxes=False):

        self.result_json_block_4 = {}

        converted_boxes = get_processed_boxes_and_words(img_path=block4_text_image_path,
                                                        block='block_4',
                                                        det_arch=self.hyperparameters['det_arch'],
                                                        reco_arch=self.hyperparameters['reco_arch'],
                                                        pretrained=self.hyperparameters['pretrained'],
                                                        verbose=verbose)



        converted_boxes = postprocess_boxes_and_words_arval_classic_restitution(converted_boxes,
                                                                                block='block_4',
                                                                                verbose=verbose,
                                                                                safe=True)
        if plot_boxes:
            file_dimensions = get_image_dimensions(block4_text_image_path)
            plot_boxes_with_text(converted_boxes, file_dimensions)


        for key_word in self.template['block_4']:
            if verbose:
                logger.info(f'Running {key_word}')
            self.result_json_block_4[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                      distance_margin=self.hyperparameters
                                                                          ['distance_margin'],
                                                                      max_distance=self.hyperparameters['max_distance'],
                                                                      minimum_overlap=self.hyperparameters
                                                                          ['minimum_overlap'],
                                                                      verbose=verbose)

            # if has_found_box(self.result_json_block_4[key_word]):
            if isinstance(self.result_json_block_4[key_word], dict):
                self.result_json_block_4[key_word] = self.result_json_block_4[key_word]['next'][1]


    def analyze_block2_signature_and_stamp(self,block_2_sign_path):
        raise NotImplementedError

    def analyze_block4_signature_and_stamp(self,block_4_sign_path):
        raise NotImplementedError

    def analyze_block4(self):
        #We check if the block4 is subdvided:
        if hasattr(self, "block_4_info_path") and hasattr(self, "block_4_sign_path"):
            self.analyze_block4_text(self.block_4_info_path, verbose=False, plot_boxes=False)
            self.analyze_block4_signature_and_stamp(self.block_4_sign_path)
        else:
            self.analyze_block4_text(self.file_path_block4, verbose=False, plot_boxes=False)
            self.analyze_block4_signature_and_stamp(self.file_path_block4)

    def analyze_block2(self):
        #We check if the block2 is subdvided:
        if hasattr(self, "block_2_info_path") and hasattr(self, "block_2_sign_path"):
            self.analyze_block2_text(self.block_2_info_path, verbose=False, plot_boxes=False)
            self.analyze_block2_signature_and_stamp(self.block_2_sign_path)
        else:
            self.analyze_block2_text(self.file_path_block2, verbose=False, plot_boxes=False)
            self.analyze_block2_signature_and_stamp(self.file_path_block2)

    def manage_orientation(self):
        """
        Check if the image is rotated and rotate it if needed.
        """
        try:
            self.document_orientation = get_image_orientation(self.path_to_document)
            if self.document_orientation != 'Horizontal':
                logger.info(f'Rotating {self.document_name}...')
                self.path_to_document = rotate_image(self.path_to_document)
        except Exception as e:
            logger.error(f'An error occurred trying to rotate {self.document_name}:{e}')


    def assess_overall_quality(self):
        logger.info(f'Assessing overall quality...')
        logger.info("Overall quality asssessment is not implemented yet in the custom pipeline")

    def analyze(self):
        logger.info(f'Analyzing {self.document_name}')
        # self.manage_orientation()
        self.assess_overall_quality()
        self.get_or_create_blocks()
        logger.info(f'Getting result template...')
        self.get_result_template()
        logger.info(f'Analyzing block 2...')
        self.analyze_block2()
        logger.info(f'Analyzing block 4...')
        self.analyze_block4()

        self.results = {}
        self.results['File Name'] = self.document_name
        self.results['block_2'] = self.result_json_block_2
        self.results['block_4'] = self.result_json_block_4

    def save_results(self):
        self.results_json = json.dumps(self.results)


class ArvalClassicGPTDocumentAnalyzer(ArvalClassicDocumentAnalyzer):

    def analyze_block4_text(self,block4_text_image_path, verbose=False, plot_boxes=False):
        if plot_boxes:
            image = PIL.Image.open(block4_text_image_path)
            plt.figure(figsize=(15, 15))
            plt.imshow(image)
            plt.show()

        # Plot the block4
        payload = build_block_checking_payload(keys=self.template['block_4'],
                                               image_path=block4_text_image_path)

        response = request_completion(payload)
        self.result_json_block_4 = json.loads(response["choices"][0]['message']['content'])

    def analyze_block2_text(self,block2_text_image_path, verbose=False, plot_boxes=False):
        # self.block_2_info_path = "/Users/amielsitruk/work/terra_cognita/customers/pop_valet/ai_documents/data/performances_data/valid_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/blocks/DM-984-VT_Proces_verbal_de_restitution_page-0001_block 2.png"
        if plot_boxes:
            image = PIL.Image.open(block2_text_image_path)
            plt.figure(figsize=(15, 15))
            plt.imshow(image)
            plt.show()

        payload = build_block_checking_payload(keys=self.template['block_2'],
                                               image_path=block2_text_image_path)


        response = request_completion(payload)
        self.result_json_block_2 = json.loads(response["choices"][0]['message']['content'])

    def assess_overall_quality(self):
        payload = build_overall_quality_checking_payload(image_path=self.path_to_document)
        response = request_completion(payload)
        self.overall_quality = response["choices"][0]['message']['content']
    def analyze_block2_signature_and_stamp(self,block_2_sign_path):
        logger.info(f'Analyzing block 2 signature and stamp...')
        payload = build_signature_checking_payload(image_path=block_2_sign_path)
        response = request_completion(payload)
        self.signature_and_stamp_2 = response["choices"][0]['message']['content']
    def analyze_block4_signature_and_stamp(self,block_4_sign_path):
        logger.info(f'Analyzing block 2 signature and stamp...')

        payload = build_signature_checking_payload(image_path=block_4_sign_path)
        response = request_completion(payload)
        self.signature_and_stamp_4 = response["choices"][0]['message']['content']
    def analyze(self):
        super().analyze()
        self.results['overall_quality'] = self.overall_quality
        self.results['signature_and_stamp_block_2'] = self.signature_and_stamp_2
        self.results['signature_and_stamp_block_4'] = self.signature_and_stamp_4
