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
    

# +
import pandas as pd
import os
import cv2
import re
import json
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
# %load_ext autoreload
# %autoreload 2

#Importing functions
from template_matching_function import get_image_dimensions,sam_pre_template_matching_function,\
    draw_contour_rectangles_on_image,crop_blocks_in_image, arval_classic_divide_and_crop_block2, arval_classic_divide_and_crop_block4,\
    find_top_and_bot_of_arval_classic_restitution,resize_arval_classic,get_bloc2_rectangle,get_bloc4_rectangle,draw_rectangles_and_save
from pipeline import get_processed_boxes_and_words,postprocess_boxes_and_words_arval_classic_restitution
from document_parsing import find_next_right_word
from image_processing import get_image_orientation, rotate_image
from performance_estimation import has_found_box
from plotting import plot_boxes_with_text
from Levenshtein import distance as l_distance

from utils import get_result_template, clean_listdir
from pathlib import Path

class ArvalClassicDocumentAnalyzer:
    def __init__(self, document_name, path_to_document, hyperparameters):
        self.document_name = document_name
        self.path_to_document = path_to_document
        self.results = {} #To be filled with results of analysis
        self.folder_path = os.path.dirname(self.path_to_document) #Folder where the file is 
        self.tmp_folder_path = os.path.join(self.folder_path, "tmp") #Folder where we'll store the blocks
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
    
    def test_block_existence(self):
        """
        Test if the blocks already in self.tmp_folder_path.
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
        
    def read_block_path(self):
        """
        Create the blocks path attribute
        """
        file_name = ['block_2_info', 'block_2_sign', 'block_4_info', 'block_4_sign']
    
        for file_name in file_name:
            path = os.path.join(self.tmp_folder_path, f"{os.path.splitext(self.document_name)[0]}_{file_name}.jpeg")
            setattr(self, f"{file_name}_path", path)      
    
    def apply_block_cropping(self):
        """
        Divide the arval_classic_restitution type document in 4 parts and save them in self.tmp_folder_path.
        """


        try:
            # Temporary file:
            logger.info("Using SAM to crop image...")
            output_temp_file_sam = str(self.tmp_folder_path) + '/SAM_' + self.document_name
            sam_pre_template_matching_function(str(self.path_to_document), output_temp_file_sam, plot_option=False)

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

            #draw_contour_rectangles_on_image(str(self.path_to_document), [block2, block4])
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
        except Exception as e:
            logger.error(f"An error occurred trying to crop the image {self.document_name}:{e}")

        # Dividing and cropping block 2 in sub-blocks:
        try:
            logger.info("Dividing block 2...")
            file_path_block2 = str(cropped_image_paths[0])
            self.block_2_info_path, self.block_2_sign_path = arval_classic_divide_and_crop_block2(file_path_block2,
                                                                                                 self.tmp_folder_path,
                                                                                                 self.document_name,
                                                                                                 self.template_path_signature_block2
                                                                                                 )

        except Exception as e:
            logger.error(f"An error occurred trying to divide block 2 in two {self.document_name}:{e}")
        
        # Dividing and cropping block 4 in sub-blocks:
        try:
            logger.info("Dividing block 4...")
            file_path_block4 = str(cropped_image_paths[1])
            self.block_4_info_path, self.block_4_sign_path = arval_classic_divide_and_crop_block4(file_path_block4,
                                                                                                self.tmp_folder_path,
                                                                                                self.document_name,
                                                                                                self.template_path_signature_block4
                                                                                                )
        except Exception as e:
            logger.error(f"An error occurred trying to divide block 4 in two {self.document_name}:{e}")

    def get_blocks(self):
        """
        Get the blocks: Create them if they don't exist or just retrieve them if they're already in self.tmp_folder_path
        """
        if self.test_block_existence():
            self.read_block_path()
        else:
            self.apply_block_cropping()

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
        
    def analyze_block2_text(self, verbose=False, plot_boxes=False):

        self.result_json_block_2 = {}
        
        converted_boxes = get_processed_boxes_and_words(img_path=self.block_2_info_path,
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
            file_dimensions = get_image_dimensions(self.block_2_info_path)
            plot_boxes_with_text(converted_boxes, file_dimensions)
        
        for key_word in self.template['block_2']:
                if verbose:
                    logger.info(f'Running {key_word}')
                self.result_json_block_2[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                 distance_margin=self.hyperparameters['distance_margin'],
                                                                 max_distance=self.hyperparameters['max_distance'],
                                                                 minimum_overlap=self.hyperparameters['minimum_overlap'],
                                                                 verbose=verbose)

                if isinstance(self.result_json_block_2[key_word], dict):
                    self.result_json_block_2[key_word] = self.result_json_block_2[key_word]['next'][1]
        
        

    def analyze_block4_text(self,verbose=False,plot_boxes=False):

        self.result_json_block_4 = {}
        
        converted_boxes = get_processed_boxes_and_words(img_path=self.block_4_info_path,
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
            file_dimensions = get_image_dimensions(self.block_4_info_path)
            plot_boxes_with_text(converted_boxes, file_dimensions)
        
                    
        for key_word in self.template['block_4']:
                if verbose:
                    logger.info(f'Running {key_word}')
                self.result_json_block_4[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                 distance_margin=self.hyperparameters['distance_margin'],
                                                                 max_distance=self.hyperparameters['max_distance'],
                                                                 minimum_overlap=self.hyperparameters['minimum_overlap'],
                                                                 verbose=verbose)
            
                
                #if has_found_box(self.result_json_block_4[key_word]):
                if isinstance(self.result_json_block_4[key_word], dict):
                    self.result_json_block_4[key_word] = self.result_json_block_4[key_word]['next'][1]
                    

    def analyze_block2_signature(self):
        raise NotImplementedError

    def analyze_block4_signature(self):
        raise NotImplementedError

    def manage_orientation(self):
        """
        Check if the image is rotated and rotate it if needed.
        """
        try:
            self.document_orientation = get_image_orientation(self.document_path)
            if self.document_orientation != 'Horizontal':
                logger.info(f'Rotating {self.document_name}...')
                self.document_path = rotate_image(self.document_path)
        except Exception as e:
            logger.error(f'An error occurred trying to rotate {self.document_name}:{e}')



    def analyze(self):
        logger.info(f'Analyzing {self.document_name}')
        #self.manage_orientation()
        logger.info(f'Getting blocks...')
        self.get_blocks()
        logger.info(f'Getting result template...')
        self.get_result_template()
        logger.info(f'Analyzing block 2...')
        self.analyze_block2_text(verbose=False, plot_boxes=False)
        logger.info(f'Analyzing block 4...')
        self.analyze_block4_text(verbose=False, plot_boxes=False)
        #self.analyze_block2_signature()
        #self.analyze_block4_signature()

        self.results = {}
        self.results['File Name'] = self.document_name
        self.results['block_2'] = self.result_json_block_2
        self.results['block_4'] = self.result_json_block_4

        self.results_json = json.dumps(self.results)


class ResultValidator:
    def __init__(self, results, plate_number):
        #with open(result_json) as f:
         #   self.result = json.load(f)
        self.result = results 
        self.signature_is_ok = True
        self.stamp_is_ok = True
        self.mileage_is_ok = True
        self.number_plate_is_filled = True
        self.number_plate_is_right = True
        self.block4_is_filled = True
        self.block4_is_filled_by_company = True
        self.plate_number = plate_number

    def validate_signatures(self):
        raise NotImplementedError

    def validate_stamp(self):
        raise NotImplementedError

    def validate_mileage(self):
        self.mileage_is_ok = self.result['block_2']['Kilométrage'].isdigit()

    def validate_number_plate_is_filled(self):
        self.number_plate_is_filled = self.result['block_2']['Immatriculé'] != "<EMPTY>"

    def validate_number_plate_is_right(self):
        detected_plate_number = self.result['block_2']['Immatriculé']
        self.number_plate_is_right = l_distance(detected_plate_number, self.plate_number) < 3
        
    def validate_block4_is_filled_by_company(self, distance_margin=4):
        company_name = self.result['block_4']['Société']
        self.block4_is_filled_by_company = company_name not in ["<EMPTY>", "<NOT_FOUND>"] \
                                           and l_distance(company_name, "Pop Valet") > distance_margin


    def validate_block4_is_filled(self):
        #TODO: Check how we want to define this function
        self.block4_is_filled = any(
            value not in ["<EMPTY>", "<NOT_FOUND>"] for value in self.result['block_4'].values()
            )

        
        

    def gather_refused_motivs(self):
        # Initialize an empty list to store the names of variables that are False
        self.refused_causes = []

        # Check each variable and add its name to the list if it's False
        #if not self.signature_is_ok:
        #   self.refused_causes.append('signature_is_ok')
        #if not self.stamp_is_ok:
        #    self.refused_causes.append('stamp_is_ok')
        if not self.mileage_is_ok:
            self.refused_causes.append('mileage_is_not_ok')
        if not self.number_plate_is_filled:
            self.refused_causes.append('number_plate_is_not_filled')
        if not self.number_plate_is_right:
            self.refused_causes.append('number_plate_is_not_right')
        if not self.block4_is_filled:
           self.refused_causes.append('block4_is_not_filled')
        if not self.block4_is_filled_by_company:
           self.refused_causes.append('block4_is_not_filled_by_company')

        

    def validate(self):
        #self.validate_signatures()
        #self.validate_stamp()
        self.validate_mileage()
        self.validate_number_plate_is_filled()
        self.validate_number_plate_is_right()        
        self.validate_block4_is_filled()
        self.validate_block4_is_filled_by_company()
        self.gather_refused_motivs()

        self.validated = self.signature_is_ok and self.stamp_is_ok and self.mileage_is_ok and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled and self.block4_is_filled_by_company
        return self.validated


invalid_restitutions_infos = pd.read_csv('data/arval/links_to_dataset/invalid_restitutions.csv')

#Getting all the documents path and name
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
all_documents = {}
for status in [#'valid',
            'invalid'
               ]:
    image_directory = Path(f'data/performances_data/{status}_data/arval_classic_restitution_images/')
    image_files = os.listdir(image_directory)

    # Iterate over each image and perform the operations
    for file_name in image_files:
        # Check if the file is an image
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            file_path = str(image_directory / file_name)
            all_documents[file_name] = {}
            all_documents[file_name]['path'] = file_path
            all_documents[file_name]['validated'] = (status == 'valid')
            if status == "valid":
                all_documents[file_name]['cause'] = "-"
            else:
                all_documents[file_name]['cause'] = invalid_restitutions_infos.loc[invalid_restitutions_infos['plateNumber'].apply(lambda x: x in file_name) & invalid_restitutions_infos['filename'].apply(lambda x: os.path.splitext(x.replace(' ', '_'))[0] in file_name)].values[0]

            all_documents[file_name]['plate_number'] = file_name.split('_')[0]

# +
#random hyper parameter: 
hyperparameters = {'det_arch': "db_resnet50",
                    'reco_arch': "crnn_mobilenet_v3_large",
                    'pretrained': True,
                    'distance_margin': 5,  # find_next_right_word for words_similarity
                    'max_distance':  400,  # find_next_right_word
                    'minimum_overlap': 10  # find_next_right_word for _has_overlap
                   }


# +
# Analyze all documents and compare with the ground truth
full_result_analysis = pd.DataFrame(columns=['document_name', 'true_status', 'predicted_status', 'true_cause', 'predicted_cause'])
#for name, info in all_documents.items():

files_to_test = ['ES-337-RE_PVR.jpeg', # Block 2 is badly detected
                 'EZ-542-KH_pv_reprise.jpeg', # Block 2 is badly detected, block 4 is not detected
                 'FB-568-VP_ARVAL_PV.jpeg',
                 'FF-173-LL_PV_restitution.jpeg', # Blocks 2 and 4 are not detected
                 'FF-404-LL_Pv_de_restitution.jpeg', # Blocks 2 and 4 are not detected
                 'FK-184-AJ_PV_de_restitution.png', # Block 2 badly detected and block 4 badly separated
                 'FS-127-LS_PV_ARVAL.jpeg', # Blocks 2 and 4 are note detected
                 'GB-587-GR_PV_DE_RESTITUTION_GB-587-GR.jpeg',
                 'GJ-053-HN_PV_Arval.jpeg' # Blocks 2 and 4 are not detected
                ]

i =0
files_iterable = {file: all_documents[file] for file in files_to_test}.items()
for name, info in files_iterable:

    try:
        document_analyzer = ArvalClassicDocumentAnalyzer(name, info['path'], hyperparameters)
        document_analyzer.analyze()
        #document_analyzer.plot_blocks()
    
        result_validator = ResultValidator(document_analyzer.results, plate_number=info['plate_number'])
        result_validator.validate()

        full_result_analysis = pd.concat([full_result_analysis,
            pd.DataFrame({
            'document_name': [name],
            'true_status': [info['validated']],
            'predicted_status': [result_validator.validated],
            'true_cause': [info['cause']],
            'predicted_cause': [", ".join(result_validator.refused_causes)],
            'details': [document_analyzer.results],
            }, index=[0])
            ])
        i += 1
    except Exception as e:
        logger.error(f"Error while analyzing {name}")


#full_result_analysis.to_csv('data/performances_data/full_result_analysis.csv', index=False)



files_iterable = {file: all_documents[file] for file in files_to_test}.items()


#Test on valid Files
files_to_test = clean_listdir(Path('data/performances_data/valid_data/arval_classic_restitution_images/'))
print(files_to_test)
for name in files_to_test:
    pathtofile = 'data/performances_data/valid_data/arval_classic_restitution_images/'+ name
    document_analyzer = ArvalClassicDocumentAnalyzer(name, pathtofile, hyperparameters)
    document_analyzer.get_blocks()
    #document_analyzer.plot_blocks()

    i += 1


print(' ')
print('Number of file analysed :',i)
print(' ')