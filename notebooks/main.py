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

# +

import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from tqdm import tqdm
from document_analysis import ArvalClassicDocumentAnalyzer,ArvalClassicGPTDocumentAnalyzer


from Levenshtein import distance as l_distance
from loguru import logger

# %load_ext autoreload
# %autoreload 2



from utils import get_result_template, clean_listdir



class ResultValidator:
    def __init__(self, results, plate_number):
        #with open(result_json) as f:
         #   self.result = json.load(f)
        self.result = results
        self.quality_is_ok = True # Only for ArvalClassicGPTDocumentAnalyzer
        self.signatures_are_ok = True
        self.stamps_are_ok = True
        self.mileage_is_ok = True
        self.number_plate_is_filled = True
        self.number_plate_is_right = True
        self.block4_is_filled = True
        self.block4_is_filled_by_company = True
        self.plate_number = plate_number

    def validate_quality(self):
        self.quality_is_ok = self.result['overall_quality'].lower() == 'yes'
    def validate_signatures(self):
        signature_block_2 = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        signature_block_4 = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if signature_block_2 and signature_block_4:
            self.stamps_are_ok = True
        else:
            self.stamps_are_ok = False

    def validate_stamps(self):
        stamp_block_2 = self.result['signature_and_stamp_block_2'] in ('both', 'signature')
        stamp_block_4 = self.result['signature_and_stamp_block_4'] in ('both', 'signature')

        if stamp_block_2 and stamp_block_4:
            self.signatures_are_ok = True
        else:
            self.signatures_are_ok = False

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
        if not self.quality_is_ok:
            self.refused_causes.append('quality_is_not_ok')
        if not self.signatures_are_ok:
           self.refused_causes.append('signature_is_ok')
        if not self.stamps_are_ok:
            self.refused_causes.append('stamp_is_ok')
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
        self.validate_quality()
        self.validate_signatures()
        self.validate_stamps()
        self.validate_mileage()
        self.validate_number_plate_is_filled()
        self.validate_number_plate_is_right()        
        self.validate_block4_is_filled()
        self.validate_block4_is_filled_by_company()
        self.gather_refused_motivs()

        self.validated = self.stamps_are_ok and self.stamps_are_ok and self.mileage_is_ok and self.number_plate_is_filled and self.number_plate_is_right and self.block4_is_filled and self.block4_is_filled_by_company
        return self.validated

normalize_str = lambda s: ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

invalid_restitutions_infos = pd.read_csv('data/links_to_dataset/invalid_restitutions.csv')
invalid_restitutions_infos['formatted_filename'] = invalid_restitutions_infos['filename'].apply(lambda x: normalize_str(os.path.splitext(x.replace(' ', ''))[0]))
#Getting all the documents path and name
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
all_documents = {}
for status in [#'valid',
               'invalid'
               ]:
    image_directory = Path(f'data/performances_data/{status}_data/arval_classic_restitution_images/')
    image_files = os.listdir(image_directory)
    image_files = [file_name for file_name in image_files if not file_name.startswith('DR-269-QA')] # I don't know why this file is here, it is not in the invalid_restitutions.csv

    # Iterate over each image and perform the operations
    for file_name in image_files:
        try:
            # Check if the file is an image
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                file_path = str(image_directory / file_name)
                all_documents[file_name] = {}
                all_documents[file_name]['path'] = file_path
                all_documents[file_name]['validated'] = (status == 'valid')
                print(file_name)

                if status == "valid":
                    all_documents[file_name]['cause'] = "-"
                else:
                    same_filename = invalid_restitutions_infos['formatted_filename'].apply(lambda x: x in normalize_str(file_name))
                    same_plate_number = invalid_restitutions_infos['plateNumber'].apply(lambda x: x in file_name)
                    all_documents[file_name]['cause'] = invalid_restitutions_infos.loc[same_filename & same_plate_number,
                    'adminComment'].values[0]
                all_documents[file_name]['plate_number'] = file_name.split('_')[0]
        except Exception as e:
            print(e)
            print(file_name)
            print(file_path)
            print(all_documents[file_name])
            print("\n\n")

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

WITH_GPT = True

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

bad_orientation_file = ["EC-609-NN_PVR.jpeg",
"EH-082-TV_PVderestitution.jpeg",
"ET-679-SV_PVrestitutionArval.jpeg",
"EZ-561-VR_PVARVAL.jpeg",
"FA-580-FY_Pvderestitution.jpeg",
"FA-772-LB_Pv.jpeg",
"FF-495-RB_20230823_101857.jpeg",
"FG-767-EX_Pvdelivraisonv.jpeg",
"FG-882-EW_PV.jpeg",
"FH-639-SE_Pvrestitution.jpeg",
"FL-147-SN_Pvarval.jpeg",
"FN-117-GQ_PVARVAL.jpeg",
"FY-915-LM_PVderestitution.jpeg",
"GB-884-EE_PV.jpeg",
"GF-784-CM_PVARVAL.jpeg"]

files_to_exclude = [] + bad_orientation_file  # Could depends on different cases

files_to_test = all_documents.keys()

files_to_iterate = {file: all_documents[file] for file in sorted(files_to_test)[:50] if file not in files_to_exclude}.items()

for name, info in tqdm(files_to_iterate):
    break
    try:
        if WITH_GPT:
            document_analyzer = ArvalClassicGPTDocumentAnalyzer(name, info['path'], hyperparameters)
        else:
            document_analyzer = ArvalClassicDocumentAnalyzer(name, info['path'], hyperparameters)
        document_analyzer.analyze()
        document_analyzer.save_results()
        #document_analyzer.plot_blocks()

        logger.info(f"Result: {document_analyzer.results}")

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
    except Exception as e:

        logger.error(f"Error {e} while analyzing {name}")


dt = datetime.now().strftime("%Y%m%d_%H%M%S")
full_result_analysis.to_csv(f'data/performances_data/full_result_analysis_{dt}.csv', index=False)

files_iterable = {file: all_documents[file] for file in files_to_test}.items()

i = 0

#Test on valid Files
path_fo_folder = Path('data/performances_data/invalid_data/arval_classic_restitution_images/')
files_to_test = clean_listdir(path_fo_folder)
files_to_test = files_to_test[1:]
print(files_to_test)
for name in files_to_test:
    path_to_file = path_fo_folder / name
    document_analyzer = ArvalClassicDocumentAnalyzer(name, path_to_file, hyperparameters)
    document_analyzer.analyze()

    #document_analyzer.plot_blocks()

    i += 1
    break

print(' ')
logger.info(f'Number of file analysed :{i}')
print(' ')