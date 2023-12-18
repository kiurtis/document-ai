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
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from tqdm import tqdm
from document_analysis import ArvalClassicDocumentAnalyzer, ArvalClassicGPTDocumentAnalyzer
from document_validation import ResultValidator
from loguru import logger


from utils import normalize_str

# %load_ext autoreload
# %autoreload 2

invalid_restitutions_infos = pd.read_csv('data/links_to_dataset/invalid_restitutions.csv')
invalid_restitutions_infos['formatted_filename'] = invalid_restitutions_infos['filename'].apply(lambda x: normalize_str(os.path.splitext(x.replace(' ', '_'))[0]))

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

        try:
            # Check if the file is an image
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                file_path = str(image_directory / file_name)
                all_documents[file_name] = {}
                all_documents[file_name]['path'] = file_path
                all_documents[file_name]['validated'] = (status == 'valid')
                all_documents[file_name]['plate_number'] = file_name.split('_')[0]


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

#files_to_exclude = [] + bad_orientation_file  # Could depends on different cases

files_to_test = all_documents.keys()
#failing_file_explained = ['EM-272-VS_Document_p1.jpeg' # Un doigt bloque la reconnaissance d'un des templates
#                 ]
#working_files = pd.read_csv('results/full_result_analysis_20231208_192447.csv')['document_name'].tolist()
#working_files += ["EN-869-YH_Pvreprise_p1.jpeg",
#                  'EC-609-NN_PVR.jpeg',
#                  'ET-679-SV_PVrestitutionArval.jpeg']
#files_to_exclude = [] + working_files + failing_file_explained

#files_to_test = [#'EQ-431-AP_pv_de_restitution_arval_X1__p1.jpeg',
                 #'ET-679-SV_PV_restitution_Arval_.jpeg',
                 #'EY-148-HE_DocumentPv_de_restitution_p1.jpeg',
                 #'EZ-542-KH_pv_reprise.jpeg',
                 #'EZ-561-VR_PV_ARVAL.jpeg',
                 #'FA-463-MX_pv_de_restitution__p1.jpeg',
                 #'FA-580-FY_Pv_de_restitution.jpeg',
                 #'FA-772-LB_Pv.jpeg',
                 #'FC-006-LG_Document_p1.jpeg',
                 #'FC-080-PV_PV_de_reprise_p1.jpeg',
                 #'FD-909-QB_pv_restitution__p1.jpeg',
                 #'FF-404-LL_Pv_de_restitution.jpeg',
                 #'FF-443-DA_PV_de_reprise_p1.jpeg',
                 #'FG-018-EB_PVR_p1.jpeg',
                 #'FG-767-EX_Pv_de_livraison_.jpeg',
                 #'FG-882-EW_PV.jpeg',
                 #'FG-926-HK_PV_de_restitution_final_ARVAL_de_FG-926-HK__p1.jpeg',
                 #'FH-639-SE_Pv_restitution.jpeg',
                 #'FH-681-LZ_ARVAL_Service_Lease_-_PV_restitution_p1.jpeg',
                 #'FJ-068-NV_PV_de_reprise_p1.jpeg',
                 #'FL-147-SN_Pv_arval.jpeg',
                 #'FM-444-ZE_pv_arval_p1.jpeg'
#                ]

'''files_to_test = [
                'EQ-431-AP_pv_de_restitution_arval_X1__p1.jpeg',
                'ES-337-RE_PVR.jpeg',
                'ET-679-SV_PV_restitution_Arval_.jpeg',
                'EZ-542-KH_pv_reprise.jpeg',
                'EZ-561-VR_PV_ARVAL.jpeg',
                'EZ-912-QS_PV_de_reprise_p2.jpeg',
                'FA-463-MX_pv_de_restitution__p1.jpeg',
                'FA-580-FY_Pv_de_restitution.jpeg',
                'FF-404-LL_Pv_de_restitution.jpeg',
                'FF-724-NB_Document_pv_p1.jpeg',
                'FG-882-EW_PV.jpeg',
                'FH-639-SE_Pv_restitution.jpeg',
                'FJ-324-KV_PV_de_reprise_p1.jpeg',
                'FJ-745-XQ_PV_de_reprise_p1.jpeg',
                'FK-184-AJ_PV_de_restitution.png',
                'FK-468-LV_PV_de_reprise_p1.jpeg',
                'FL-115-PN_Pv_de_restitution_p1.jpeg',
                 'FL-147-SN_Pv_arval.jpeg'
                 ]'''
files_to_exclude = []
files_to_iterate = {file: all_documents[file]
                    for file in sorted(files_to_test)[:50]
                    if file not in files_to_exclude}.items()

for name, info in tqdm(files_to_iterate):

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
            'error':[None]
            }, index=[0])
            ])
    except Exception as e:
        pd.concat([full_result_analysis,
                   pd.DataFrame({
                       'document_name': [name],
                       'true_status': [info['validated']],
                       'predicted_status': [None],
                       'true_cause': [info['cause']],
                       'predicted_cause': [None],
                       'details': [None],
                       'error':[e]
                   }, index=[0])
                   ])
        logger.exception(f"Error: {e} while analyzing {name}")


dt = datetime.now().strftime("%Y%m%d_%H%M%S")
full_result_analysis.to_csv(f'results/full_result_analysis_{dt}.csv', index=False)

files_iterable = {file: all_documents[file] for file in files_to_test}.items()


# Typologie d'erreur
# - Could not find block 2
# - Could not find block 4
