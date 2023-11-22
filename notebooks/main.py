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

#Importing functions
from template_matching_function import get_image_dimensions,arval_classic_create_bloc2,arval_classic_create_bloc4,draw_contour_rectangles_on_image,crop_and_save_image,arval_classic_divide_and_crop_bloc2,arval_classic_divide_and_crop_bloc4
from pipeline import get_processed_boxes_and_words,postprocess_boxes_and_words
from document_parsing import find_next_right_word
from performance_estimation import has_found_box
from plotting import plot_boxes_with_text
from Levenshtein import distance as l_distance

from utils import get_result_template
from pathlib import Path


def verify_number_plate_format(s):
    """
    Verify if a string follows a the specific French plate format number.
    """
    pattern = r'^[a-zA-Z]{2}-\d{3}-[a-zA-Z]{2}$'
    return bool(re.match(pattern, s))

class DocumentAnalyzer:
    def __init__(self, document_name, path_to_document,hyperparameters):
        self.document_name = document_name
        self.path_to_document = path_to_document
        self.results = {} #To be filled with results of analysis
        self.folder_path = os.path.dirname(self.path_to_document) #Folder where the file is 
        self.temps_folder_path = os.path.join(self.folder_path, "temps") #Folder where we'll store the blocs

        if not os.path.exists(self.temps_folder_path):
            os.makedirs(self.temps_folder_path)
    
    def test_bloc_existence(self):
        """
        Test if the blocs already in self.temps_folder_path.
        """
        bloc_doc = []
        file_name = ['bloc_2_info','bloc_2_sign','bloc_4_info','bloc_4_sign']
        missing_files = []

        for i in file_name:
            cropped_image_path = os.path.join(self.temps_folder_path, f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg")

            if os.path.exists(cropped_image_path):
                bloc_doc.append(cropped_image_path)  
            else:
                missing_files.append(cropped_image_path)  

        if len(missing_files) == 0 :
            return True
        else:
            return False
        
    def read_block_path(self):
        """
        Create the blocs path attribute
        """
        file_name = ['bloc_2_info','bloc_2_sign','bloc_4_info','bloc_4_sign']
    
        for file_name in file_name:
            path = os.path.join(self.temps_folder_path, f"{os.path.splitext(self.document_name)[0]}_{file_name}.jpeg")
            setattr(self, f"{file_name}_path", path)      
    
    def crop_blocks_and_save_them(self):
        """
        Divide the arval_classic_restitution type document in 4 parts and save them in self.temps_folder_path.
        """
        
        #Template used to process the template matching
        template_path_top_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_le_vehicule.png'
        template_path_top_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_descriptif.png'
        template_path_bot_bloc3 = 'data/performances_data/arval_classic_restitution_images/template/template_end_block3.png'
        template_path_bot_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_barcode.png'

        #template to subdivise the bloc:
        template_path_signature_bloc2 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_2_garage.png'
        template_path_signature_bloc4 = 'data/performances_data/arval_classic_restitution_images/template/template_bloc_4_long.png'
               
        try:
            # Getting bloc 2 and 4
            new_dimensions = get_image_dimensions(self.path_to_document)
            bloc2 = arval_classic_create_bloc2(str(self.path_to_document), template_path_top_bloc2, template_path_top_bloc3, new_dimensions)
            bloc4 = arval_classic_create_bloc4(str(self.path_to_document), template_path_bot_bloc3, template_path_bot_bloc4, new_dimensions)
            #draw_contour_rectangles_on_image(self.path_to_document, [bloc2, bloc4])
            blocs = [bloc2, bloc4]
            #print(blocs)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to get bloc 2 and 4 of ", self.document_name ," :", e)
            print('-----------------')
                
        try:
            #cropping and saving the image in bloc in the temp folder
            image = cv2.imread(self.path_to_document)
            crop_and_save_image(image, blocs, self.temps_folder_path, self.document_name)
            cropped_image_paths = [os.path.join(self.temps_folder_path, f"{os.path.splitext(self.document_name)[0]}_{i}.jpeg") for i in range(len(blocs))]
            print(cropped_image_paths)
        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to crop the image ", self.document_name ," :", e)
            print('-----------------')

        #Dividing and cropping bloc 2:
        try:
            file_path_bloc2 = str(cropped_image_paths[0])
            self.bloc_2_info_path,self.bloc_2_sign_path = arval_classic_divide_and_crop_bloc2(file_path_bloc2,self.temps_folder_path,self.document_name,template_path_signature_bloc2)

        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 2 in two", self.document_name ," :", e)
            print('-----------------')
        
        #Dividing and cropping bloc 4:
        try:
            file_path_bloc4 = str(cropped_image_paths[1])
            self.bloc_4_info_path,self.bloc_4_sign_path = arval_classic_divide_and_crop_bloc4(file_path_bloc4,self.temps_folder_path,self.document_name,template_path_signature_bloc4)

        except Exception as e:
            print('-----------------')
            print("An error occurred tyring to divide bloc 4 in two", self.document_name ," :", e)
            print('-----------------')
              
    def get_blocks(self):
        """
        Get the blocs: Create them if they don't exist or just find them if they're already in self.temps_folder_path
        """
        if self.test_bloc_existence() == True:
            self.read_block_path()
        else:
            self.crop_blocks_and_save_them() 

    def print_blocks(self):
        for i in [self.bloc_2_info_path,self.bloc_2_sign_path,self.bloc_4_info_path,self.bloc_4_sign_path]:
            image_path = i  # Replace with the path to your image
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.axis('off')  # Turn off axis numbers
            plt.show()

    def get_result_template(self):
        folder_ground_truths = Path('data/performances_data/arval_classic_restitution_json/')
        self.template = get_result_template(folder_ground_truths)
        
    def analyze_block2_text(self,verbose=False,plot_boxes=False):

        self.result_json_block_2 = {}
        
        converted_boxes = get_processed_boxes_and_words(img_path=self.bloc_2_info_path,
                                                        block='block_2',
                                                        det_arch=hyperparameters['det_arch'],
                                                        reco_arch=hyperparameters['reco_arch'],
                                                        pretrained=hyperparameters['pretrained'],
                                                        verbose=verbose)

        if plot_boxes:
            plot_boxes_with_text(converted_boxes)
        
        converted_boxes = postprocess_boxes_and_words(converted_boxes,
                                                          block='block_2',
                                                          verbose=verbose,
                                                          safe=True)
        
        for key_word in self.template['block_2']:
                if verbose:
                    print(f'Running {key_word}')
                self.result_json_block_2[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                 distance_margin=hyperparameters['distance_margin'],
                                                                 max_distance=hyperparameters['max_distance'],
                                                                 minimum_overlap=hyperparameters['minimum_overlap'],
                                                                 verbose=verbose)

                if isinstance(self.result_json_block_2[key_word], dict):
                    self.result_json_block_2[key_word] = self.result_json_block_2[key_word]['next'][1]
        
        

    def analyze_block4_text(self,verbose=False,plot_boxes=False):

        self.result_json_block_4 = {}
        
        converted_boxes = get_processed_boxes_and_words(img_path=self.bloc_4_info_path,
                                                        block='block_4',
                                                        det_arch=hyperparameters['det_arch'],
                                                        reco_arch=hyperparameters['reco_arch'],
                                                        pretrained=hyperparameters['pretrained'],
                                                        verbose=verbose)

        if plot_boxes:
            plot_boxes_with_text(converted_boxes)

        converted_boxes = postprocess_boxes_and_words(converted_boxes,
                                                          block='block_4',
                                                          verbose=verbose,
                                                          safe=True)
        
                    
        for key_word in self.template['block_4']:
                if verbose:
                    print(f'Running {key_word}')
                self.result_json_block_4[key_word] = find_next_right_word(converted_boxes, key_word,
                                                                 distance_margin=hyperparameters['distance_margin'],
                                                                 max_distance=hyperparameters['max_distance'],
                                                                 minimum_overlap=hyperparameters['minimum_overlap'],
                                                                 verbose=verbose)
            
                
                #if has_found_box(self.result_json_block_4[key_word]):
                if isinstance(self.result_json_block_4[key_word], dict):
                    self.result_json_block_4[key_word] = self.result_json_block_4[key_word]['next'][1]
                    

    def analyze_block2_signature(self):
        raise NotImplementedError

    def analyze_block4_signature(self):
        raise NotImplementedError


        
    def analyze(self):
        self.get_blocks()
        self.get_result_template()
        self.analyze_block2_text(verbose=False,plot_boxes=True)
        self.analyze_block4_text(verbose=False,plot_boxes=True)
        #self.analyze_block2_signature()
        #self.analyze_block4_signature()

        self.results = {}
        self.results['File Name'] = self.document_name
        self.results['block_2'] = self.result_json_block_2
        self.results['block_4'] = self.result_json_block_4

        self.results_json = json.dumps(self.results)


class ResultValidator:
    def __init__(self, results):
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


    def validate_signatures(self):
        raise NotImplementedError

    def validate_stamp(self):
        raise NotImplementedError

    def validate_mileage(self):
        self.mileage_is_ok = self.result['block_2']['Kilométrage'].isdigit()

    def validate_number_plate_is_filled(self):
        self.number_plate_is_filled = self.result['block_2']['Immatriculé'] != "<EMPTY>"

    def validate_number_plate_is_right(self):
        plate_number= self.result['block_2']['Immatriculé']
        self.number_plate_is_right = verify_number_plate_format(plate_number)
        
    def validate_block4_is_filled_by_company(self):
        company_name = self.result['block_4']['Société']
        self.block4_is_filled_by_company = company_name not in ["<EMPTY>", "<NOT_FOUND>"] and l_distance(company_name, "Pop Valet") > 4


    def validate_block4_is_filled(self):
        #TO DO: Check how we want to define this function
        self.block4_is_filled = any(
            value not in ["<EMPTY>", "<NOT_FOUND>"] for value in self.result['block_4'].values()
            )

        
        

    def gather_refused_motivs(self):
        # Initialize an empty list to store the names of variables that are False
        self.refused_causes = []

        # Check each variable and add its name to the list if it's False
        #if not self.signature_is_ok:
        #   self.refused_motiv.append('signature_is_ok')
        #if not self.stamp_is_ok:
        #    self.refused_motiv.append('stamp_is_ok')
        if not self.mileage_is_ok:
            self.refused_causes.append('mileage_is_ok')
        if not self.number_plate_is_filled:
            self.refused_causes.append('number_plate_is_filled')
        if not self.number_plate_is_right:
            self.refused_causes.append('number_plate_is_right')
        if not self.block4_is_filled:
           self.refused_causes.append('block4_is_filled')
        if not self.block4_is_filled_by_company:
           self.refused_causes.append('block4_is_filled_by_company')

        

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

# +
documents = {'EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg': {'path':'data/performances_data/arval_classic_restitution_images/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_PUBLIC_LLD_p1.jpeg',
                           'validated': False,
                           'cause': 'block4_is_filled_by_company'},
             'document2': {'path':'path/to/document2',
                           'validated': False,
                           'cause': 'block4_is_filled_by_company'}
             }

#Getting all the documents path and name
all_documents = {}

image_directory = Path('data/performances_data/arval_classic_restitution_images/')
image_files = os.listdir(image_directory)

# Iterate over each image and perform the operations
valid_image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']

# Iterate over each image and perform the operations
for file_name in image_files:
    # Check if the file is an image
    if any(file_name.lower().endswith(ext) for ext in valid_image_extensions):
        file_path = str(image_directory / file_name)
        all_documents[file_name] = {}
        all_documents[file_name]['path']=file_path
        all_documents[file_name]['validated']=False
        all_documents[file_name]['cause']='block4_is_filled_by_company'


# +
#random hyper parameter: 
hyperparameters = {'det_arch':"db_resnet50",
        'reco_arch':"crnn_mobilenet_v3_large",
        'pretrained':True ,
        'distance_margin': 10, # find_next_right_word for words_similarity
        'max_distance':  300, # find_next_right_word
        'minimum_overlap': 20 # find_next_right_word for _has_overlap
}

full_result_analysis = pd.DataFrame(columns=['document_name', 'true_status', 'predicted_status', 'true_cause', 'predicted_cause'])
for name, info in all_documents.items():
    document_analyzer = DocumentAnalyzer(name, info['path'],hyperparameters)
    document_analyzer.analyze()
    print(document_analyzer.results)
    result_validator = ResultValidator(document_analyzer.results)
    result_validator.validate()

    full_result_analysis = pd.concat([full_result_analysis,
        pd.DataFrame({
        'document_name': [name],
        'true_status': [info['validated']],
        'predicted_status': [result_validator.validated],
        'true_cause': [info['cause']],
        'predicted_cause': [", ".join(result_validator.refused_causes)]
        }, index=[0])
        ])
    document_analyzer.print_blocks()
    #print(result_validator.validated)
    break

print(full_result_analysis)
# -

document_analyzer.print_blocks()

print(full_result_analysis)

# +
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_images("data/performances_data/arval_classic_restitution_images/test_signa/EK-744-NX_EK-744-NX_procès_verbal_de_restitution_définitive_Arval_exemplaire_locataire client_p1_bloc_2_info.jpeg")
# Analyze
result = model(doc)



# -

result.show(doc)


