import PIL
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
from loguru import logger
from pathlib import Path


#Importing functions
from ai_documents.utils import get_result_template, has_non_none_attributes
from ai_documents.detection.template_matching import get_image_dimensions, crop_blocks_in_image, \
    arval_classic_divide_and_crop_block2, arval_classic_divide_and_crop_block4,\
    find_top_and_bot_of_arval_classic_restitution, resize_arval_classic, get_block2_rectangle, get_block4_rectangle, \
    draw_rectangles_and_save
from ai_documents.detection.sam import sam_pre_template_matching_function
from ai_documents.analysis.cv.boxes_processing import get_processed_boxes_and_words,\
    postprocess_boxes_and_words_arval_classic_restitution
from ai_documents.analysis.cv.document_parsing import find_next_right_word
from ai_documents.analysis.lmm.gpt import build_block_checking_payload, request_completion, \
    build_overall_quality_checking_payload, build_signature_checking_payload, number_plate_check_gpt, \
    build_block4_checking_payload
from ai_documents.plotting import plot_boxes_with_text
from ai_documents.exceptions import DocumentAnalysisError, LMMProcessingError, BlockDetectionError


class ArvalClassicDocumentAnalyzer:
    def __init__(self, document_name, path_to_document, hyperparameters=None):
        self.document_name = document_name
        self.path_to_document = path_to_document
        self.results = {}  # To be filled with results of analysis
        self.results['details'] = {}

        self.folder_path = Path(self.path_to_document).parent  # Folder where the file is
        self.tmp_folder_path = self.folder_path / "tmp" / self.document_name.split(".")[0] # Folder where we'll store the blocks
        self.hyperparameters = hyperparameters
        self.cropped_by_sam = False
        self.block_4_info_path = None
        self.block_4_sign_path = None
        self.block_2_info_path = None
        self.block_2_sign_path = None

        # Templates used to process the template matching
        template_folder = Path('data/performances_data/template/arval_classic_restitution')
        self.template_path_top_block1 = template_folder / 'template_top_left.png'
        self.template_path_bot_block4 = template_folder / 'template_barcode.png'
        self.template_path_top_block2 = template_folder / 'template_path_top_block2.png'
        self.template_path_top_block3 = template_folder / 'template_descriptif.png'
        self.template_path_top_block4 = template_folder / 'template_end_block3.png'

        # Templates to subdivise the bloc:
        self.template_path_signature_block2 = template_folder / 'template_block_2_garage.png'
        self.template_path_signature_block4 = template_folder / 'template_block_4_long.png'

        if not os.path.exists(self.tmp_folder_path):
            os.makedirs(self.tmp_folder_path)

    def test_blocks_existence(self, filenames):
        """
        For each block, test if it exists. If one is missing, it return False.
        """
        block_doc = []
        missing_files = []

        for i in filenames:
            cropped_image_path = os.path.join(self.tmp_folder_path, i)

            if os.path.exists(cropped_image_path):
                block_doc.append(cropped_image_path)
            else:
                missing_files.append(cropped_image_path)

        if len(missing_files) == 0:

            return True
        else:
            return False

    def read_block_2_and_4_path(self,filenames):
        """
        Create the block 2 and 4 path attributes
        """
        cropped_image_paths = []
        for image in filenames:
            cropped_image_paths.append(self.tmp_folder_path / image)

        self.file_path_block2 = str(cropped_image_paths[0])
        self.file_path_block4 = str(cropped_image_paths[1])

    def read_block2_subdivised_path(self):
        """
        Create the block 2 subdivision path attributes
        """
        file_name = ['block_2_info', 'block_2_sign']

        for file_name in file_name:
            path = os.path.join(self.tmp_folder_path, f"{file_name}.jpeg")
            setattr(self, f"{file_name}_path", path)

    def read_block4_subdivised_path(self):
        """
        Create the block 4 subddivision path attributes
        """
        file_name = ['block_4_info', 'block_4_sign']

        for file_name in file_name:
            path = self.tmp_folder_path / f"{file_name}.jpeg"
            setattr(self, f"{file_name}_path", path)


    def cropping_block2_and_4(self):
        """
        Divide the arval_classic_restitution type document in 4 parts and save them in self.tmp_folder_path.
        """

        try:
            # Temporary file:
            logger.info("Using SAM to crop image...")
            output_temp_file_sam = sam_pre_template_matching_function(self.path_to_document, self.tmp_folder_path, plot_option=False)
            self.cropped_by_sam = True
        except Exception as e:
            logger.error(f"An error occurred trying to use SAM for the document {self.document_name}:{e}")

        try:
            # Getting block 2 and 4
            # Temporary file:
            if self.cropped_by_sam:
                resize_img = resize_arval_classic(output_temp_file_sam)
            else:
                resize_img = resize_arval_classic(self.path_to_document)

            # Resizing image:
            copy_of_rezise_img = resize_img.copy()

            # Finding the bottom and the top of the document :
            top_rect, bottom_rect = find_top_and_bot_of_arval_classic_restitution(copy_of_rezise_img, self.tmp_folder_path,
                                                                                  self.template_path_top_block1,
                                                                                  self.template_path_bot_block4,
                                                                                  plot_img=False)
            copy_of_rezise_img = resize_img.copy()

            # Searching block2
            logger.info("Getting blocks 2...")
            output_temp_file = self.tmp_folder_path / 'block_2.jpeg'
            block2 = get_block2_rectangle(copy_of_rezise_img, output_temp_file, top_rect, bottom_rect,
                                         self.template_path_top_block2, self.template_path_top_block3, plot_img=False)
            logger.info("Getting blocks 4...")
            copy_of_rezise_im = resize_img.copy()

            output_temp_file = self.tmp_folder_path / 'block_4.jpeg'
            block4 = get_block4_rectangle(copy_of_rezise_im, output_temp_file, block2, bottom_rect,
                                         self.template_path_top_block4, plot_img=False)

            copy_of_rezise_im = resize_img.copy()
            draw_rectangles_and_save(copy_of_rezise_im, [block2, block4], output_temp_file)

            # draw_contour_rectangles_on_image(str(self.path_to_document), [block2, block4])
            blocks = [block2, block4]

        except Exception as e:
            raise BlockDetectionError(f"An error occurred trying to get blocks 2 and 4 of {self.document_name}: {e}")

        try:
            # Cropping and saving the blocks images in the tmp folder
            image = np.array(resize_img)
            logger.info("Cropping blocks...")

            crop_blocks_in_image(image, blocks,
                                 self.tmp_folder_path,
                                 self.document_name)
            cropped_image_paths = [os.path.join(self.tmp_folder_path,
                                                f"{os.path.splitext(self.document_name)[0]}_block_{i}.jpeg")
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
                                                                                                  self.template_path_signature_block2)
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
        blocks = []
        for i in range(2):
            blocks.append(f"{os.path.splitext(self.document_name)[0]}_block_{i}.jpeg")

        if self.test_blocks_existence(filenames= blocks): # 2 & 4
            logger.info(f'Blocks 2 and 4 already in tmp folder')
            self.read_block_2_and_4_path(filenames=blocks)
            if self.test_blocks_existence(filenames=['block_2_info.jpeg', 'block_2_sign.jpeg']): # 2 only
                logger.info(f'Blocks 2 subdivisions already in tmp folder')
                self.read_block2_subdivised_path()
            else :
                self.subdivising_and_cropping_block2()

            if self.test_blocks_existence(filenames=['block_4_info.jpeg', 'block_4_sign.jpeg']): # 4 only
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

    def get_result_template(self):
        """
        Get the template of the result json file, divided by blocks and including all the keywords.
        :return:
        """
        folder_ground_truths = Path('data/performances_data/valid_data/arval_classic_restitution_json/')
        #self.template = get_result_template(folder_ground_truths)
        self.template = {'block_2': {"Immatriculé": None,
                                         "Kilométrage": None,
                                         "Restitué le": None,
                                         "N° de série": None},
                            'block_4': {
                                   "Nom et prénom": None,
                                   "E-mail": None,
                                   "Tél": None,
                                   "Société": None}
                         }
    def analyze_block2_text(self, block2_text_image_path, verbose=False, plot_boxes=False):

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

        self.results['block_2'] = self.result_json_block_2


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


        self.results['block_4'] = self.result_json_block_4

    def analyze_block2_signature_and_stamp(self,block_2_sign_path):
        raise NotImplementedError

    def analyze_block4_signature_and_stamp(self,block_4_sign_path):
        raise NotImplementedError

    def analyze_block4(self):
        #We check if the block4 is subdvided:
        if has_non_none_attributes(self, "block_4_info_path", "block_4_sign_path"):
            self.analyze_block4_text(self.block_4_info_path, verbose=False, plot_boxes=False)
            self.analyze_block4_signature_and_stamp(self.block_4_sign_path)
            self.results['details']['block_2_image_analyzed'] = 'block_4_sign_path'
        else:
            self.analyze_block4_text(self.file_path_block4, verbose=False, plot_boxes=False)
            self.analyze_block4_signature_and_stamp(self.file_path_block4)
            self.results['details']['block_4_image_analyzed'] = 'file_path_block4'


    def analyze_block2(self):
        #We check if the block2 is subdvided:
        if has_non_none_attributes(self, "block_2_info_path", "block_2_sign_path"):
            self.analyze_block2_text(self.block_2_info_path, verbose=False, plot_boxes=False)
            self.analyze_block2_signature_and_stamp(self.block_2_sign_path)
            self.results['details']['block_2_image_analyzed'] = 'block_2_sign_path'
        else:
            self.analyze_block2_text(self.file_path_block2, verbose=False, plot_boxes=False)
            self.analyze_block2_signature_and_stamp(self.file_path_block2)
            self.results['details']['block_2_image_analyzed'] = 'file_path_block2'

    def assess_overall_quality(self):
        logger.info(f'Assessing overall quality...')
        logger.info("Overall quality asssessment is not implemented yet in the custom pipeline")

    def analyze(self):
        try:
            logger.info(f'Analyzing {self.document_name}')
            self.assess_overall_quality()
            self.get_or_create_blocks()
            logger.info(f'Getting result template...')
            self.get_result_template()
            logger.info(f'Analyzing block 2...')
            self.analyze_block2()
            logger.info(f'Analyzing block 4...')
            self.analyze_block4()


        except Exception as e:
            raise DocumentAnalysisError(f'Could not analyze {self.document_name}') from e

    def save_results(self):
        self.results_json = json.dumps(self.results)


class ArvalClassicGPTDocumentAnalyzer(ArvalClassicDocumentAnalyzer):

    @staticmethod
    def safe_process_response(response, attribute):
        if 'error' not in response.keys():
            try:
                content = response["choices"][0]['message']['content']
                try:
                    content = json.loads(content)
                except:
                    pass  # content is not a string dictionary, so it is already a string and no more processing to do
                return content
            except Exception as e:
                raise LMMProcessingError(f'Could not process {attribute} response: {e}')
                #logger.warning(f'Could not process {attribute} response: {e}')
                #return None
        else:
            raise LMMProcessingError(f'Could not process {attribute} response: {response["error"]["code"]}')
            #logger.warning(f'Could not process {attribute} response: {response["error"]["code"]}')
            #return None

    def analyze_block4_text(self, block4_text_image_path, verbose=False, plot_boxes=False):
        if plot_boxes:
            image = PIL.Image.open(block4_text_image_path)
            plt.figure(figsize=(15, 15))
            plt.imshow(image)

        # Plot the block4
        #payload = build_block_checking_payload(keys=self.template['block_4'],
        #                                       image_path=block4_text_image_path)
        payload = build_block4_checking_payload(image_path=block4_text_image_path)

        response = request_completion(payload)
        logger.info(f'Block 4 response: {response}')
        self.result_json_block_4 = self.safe_process_response(response, 'result_json_block_4')
        if self.result_json_block_4 is None:
            self.result_json_block_4 = {'block_4': {'Nom et prénom': '<NOT_FOUND>',
                                                    'E-mail': '<NOT_FOUND>',
                                                    'Tél': '<NOT_FOUND>',
                                                    'le': '<NOT_FOUND>',
                                                    'Société': '<NOT_FOUND>'}}

        self.results['block_4'] = self.result_json_block_4

    def analyze_block2_text(self, block2_text_image_path, verbose=False, plot_boxes=False):
        logger.info(f'Analyzing block 2 text...')
        logger.info(f'{block2_text_image_path}')
        # self.block_2_info_path = "/Users/amielsitruk/work/terra_cognita/customers/pop_valet/ai_documents/data/performances_data/valid_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/blocks/DM-984-VT_Proces_verbal_de_restitution_page-0001_block 2.png"
        if plot_boxes:
            image = PIL.Image.open(block2_text_image_path)
            plt.figure(figsize=(15, 15))
            plt.imshow(image)

        payload = build_block_checking_payload(keys=self.template['block_2'],
                                               image_path=block2_text_image_path)


        response = request_completion(payload)
        self.result_json_block_2 = self.safe_process_response(response, 'result_json_block_2')
        if self.result_json_block_2 is None:
            self.result_json_block_2 = {'block_2': {"Immatriculé": '<NOT_FOUND>',
                                                    "Kilométrage": '<NOT_FOUND>',
                                                    "Restitué le": '<NOT_FOUND>',
                                                    "Numéro de série": '<NOT_FOUND>'}}


        #Litle gpt hack for number_plate
        plate_number = self.document_name.split('_')[0]
        response2 = request_completion(number_plate_check_gpt(plate_number, block2_text_image_path))

        plate_number_GPT = response2["choices"][0]['message']['content']

        logger.info(f'Old plate number : {self.result_json_block_2["Immatriculé"]}')
        self.result_json_block_2["Immatriculé"] = plate_number_GPT
        logger.info(f'GPT plate number : {plate_number_GPT}')
        logger.info(f'{self.result_json_block_2["Immatriculé"]}')

        self.results['block_2'] = self.result_json_block_2

    def assess_overall_quality(self):
        payload = build_overall_quality_checking_payload(image_path=self.path_to_document)
        response = request_completion(payload)
        logger.info(f'Overall quality response: {response}')
        self.overall_quality = self.safe_process_response(response, 'overall_quality')
        self.results['overall_quality'] = self.overall_quality

    def analyze_block2_signature_and_stamp(self, block_2_sign_path):
        logger.info(f'Analyzing block 2 signature and stamp...')
        logger.info(f'{block_2_sign_path}')
        payload = build_signature_checking_payload(image_path=block_2_sign_path)
        response = request_completion(payload)

        self.signature_and_stamp_block_2 = self.safe_process_response(response, 'signature_and_stamp_2')
        self.results['signature_and_stamp_block_2'] = self.signature_and_stamp_block_2

    def analyze_block4_signature_and_stamp(self,block_4_sign_path):
        logger.info(f'Analyzing block 4 signature and stamp...')
        logger.info(f'{block_4_sign_path}')
        payload = build_signature_checking_payload(image_path=block_4_sign_path)
        response = request_completion(payload)
        self.signature_and_stamp_block_4 = self.safe_process_response(response, 'signature_and_stamp_4')
        self.results['signature_and_stamp_block_4'] = self.signature_and_stamp_block_4
