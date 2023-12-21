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
import csv

import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd

from tqdm import tqdm
from ai_documents.analysis.entities import ArvalClassicDocumentAnalyzer, ArvalClassicGPTDocumentAnalyzer
from ai_documents.validation.entities import ResultValidator
from ai_documents.utils import normalize_str
from loguru import logger

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



# Typologie d'erreur
# - Could not find block 2
# - Could not find block 4


######



def read_csv(file_path):
    data = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            error_column = 'ground_truth' if 'ground_truth' in row else 'predicted_cause'
            if error_column in row:
                # Strip spaces and split the errors
                errors = [error.strip() for error in row[error_column].replace('\n', '').split(',')]
                data[row['document_name']] = set(errors)
            else:
                print(f"Missing '{error_column}' in row: {row}")  # For debugging
    return data

def compute_overall_metrics(ground_truth, predictions):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    all_errors = set()
    for errors in ground_truth.values():
        all_errors.update(errors)
    for errors in predictions.values():
        all_errors.update(errors)

    for doc_name, gt_errors in ground_truth.items():
        pred_errors = predictions.get(doc_name, set())
        for error in all_errors:
            if error in gt_errors and error in pred_errors:
                true_positives += 1
            elif error not in gt_errors and error not in pred_errors:
                true_negatives += 1
            elif error in pred_errors and error not in gt_errors:
                false_positives += 1
            elif error not in pred_errors and error in gt_errors:
                false_negatives += 1

    return true_positives, false_positives, true_negatives, false_negatives


def compute_metrics_per_error(ground_truth, predictions):
    metrics_per_error = {}

    # Create a set of all unique error types from both ground truth and predictions
    all_errors = set()
    for errors in ground_truth.values():
        all_errors.update(errors)
    for errors in predictions.values():
        all_errors.update(errors)

    all_errors.remove("")

    # Initialize metrics for each error
    for error_type in all_errors:
        if error_type:  # Skip empty error strings
            metrics_per_error[error_type] = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0}

    # Compute metrics
    for doc_name, gt_errors in ground_truth.items():
        pred_errors = predictions.get(doc_name, set())

        for error_type in all_errors:
            if error_type:  # Skip empty error strings
                if "quality_is_not_ok" in gt_errors and error_type != "quality_is_not_ok":
                    continue # Skip all errors except "quality_is_not_ok" if the document quality was indeed bad

                if (error_type in gt_errors) and (error_type in pred_errors):
                    metrics_per_error[error_type]['true_positive'] += 1
                elif (error_type not in gt_errors) and (error_type not in pred_errors):
                    metrics_per_error[error_type]['true_negative'] += 1
                elif (error_type in pred_errors) and (error_type not in gt_errors):
                    metrics_per_error[error_type]['false_positive'] += 1
                elif (error_type not in pred_errors) and (error_type in gt_errors):
                    metrics_per_error[error_type]['false_negative'] += 1

    for error_type, metrics in metrics_per_error.items():
        if error_type:  # Skip empty error strings
            total = metrics['true_positive'] + metrics['false_positive'] + metrics['true_negative'] + metrics[
                'false_negative']
            if total > 0:
                accuracy = (metrics['true_positive'] + metrics['true_negative']) / total
                metrics['accuracy'] = accuracy
            else:
                metrics['accuracy'] = 0  # Handling division by zero if there are no observations

    return metrics_per_error


if __name__ == '__main__':

    RUN_ANALYSIS = False
    RUN_METRICS_COMPUTATION = True

if RUN_ANALYSIS:

    # Analyze all documents and compare with the ground truth
    full_result_analysis = pd.DataFrame(
        columns=['document_name', 'true_status', 'predicted_status', 'true_cause', 'predicted_cause'])
    # for name, info in all_documents.items():

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

    files_to_test = all_documents.keys()


    files_to_exclude = []
    files_to_iterate = {file: all_documents[file]
                        for file in sorted(files_to_test)[:50]
                        if file not in files_to_exclude}.items()

    for name, info in tqdm(files_to_iterate):

        try:
            document_analyzer = ArvalClassicGPTDocumentAnalyzer(name, info['path'],
                                                                #hyperparameters
                                                                )
            document_analyzer.analyze()
            # document_analyzer.plot_blocks()
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
                                                  'error': [None]
                                              }, index=[0])
                                              ])
        except Exception as e:
            #raise e
            pd.concat([full_result_analysis,
                       pd.DataFrame({
                           'document_name': [name],
                           'true_status': [info['validated']],
                           'predicted_status': [None],
                           'true_cause': [info['cause']],
                           'predicted_cause': [None],
                           'details': [None],
                           'error': [e]
                       }, index=[0])
                       ])
            logger.error(f"Error {e} while analyzing {name}")
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_result_analysis.to_csv(f'results/full_result_analysis_{dt}.csv', index=False)

if RUN_METRICS_COMPUTATION:

    # Load ground truth and predictions
    ground_truth_path = 'results/invalid_data_ground_truth.csv'  # Replace with your actual path
    #predictions_path = 'results/predictions_amiel.csv'    # Replace with your actual path
    predictions_path = 'results/full_result_analysis_20231217_222026.csv'
    ground_truth_data = read_csv(ground_truth_path)
    predictions_data = read_csv(predictions_path)

    # Compute metrics
    tp, fp, tn, fn = compute_overall_metrics(ground_truth_data, predictions_data)
    print('Overall metrics :')
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")

    # Example usage
    tp_fp_tn_fn_per_error = compute_metrics_per_error(ground_truth_data, predictions_data)

    print('metrics per errors:')
    print(tp_fp_tn_fn_per_error)

    # Display the results
    for error, metrics in tp_fp_tn_fn_per_error.items():
        print(f"Error: {error}")
        print(f" True Positives: {metrics['true_positive']}")
        print(f" False Positives: {metrics['false_positive']}")
        print(f" True Negatives: {metrics['true_negative']}")
        print(f" False Negatives: {metrics['false_negative']}")
        print(f" Accuracy: {metrics['accuracy']:0.02f}\n")