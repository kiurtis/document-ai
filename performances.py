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

import numpy as np
import pandas as pd

from tqdm import tqdm
from ai_documents.analysis.entities import ArvalClassicDocumentAnalyzer, ArvalClassicGPTDocumentAnalyzer
from ai_documents.validation.entities import ResultValidator
from ai_documents.utils import normalize_str
from loguru import logger


# Typologie d'erreur
# - Could not find block 2
# - Could not find block 4


######
def expand_df(df, all_causes):
    expanded_data = []
    for _, row in df.iterrows():
        for cause in all_causes:
            expanded_data.append({
                'document_name': row['document_name'],
                'cause': cause,
                'ground_truth': cause in row['ground_truth'],
                'predicted_cause': cause in row['predicted_cause']
            })
    return pd.DataFrame(expanded_data)


def calculate_metrics(expanded_df, cause):
    if cause != 'quality_is_not_ok':  # In this case, we remove the bad quality documents
        good_quality_condition = expanded_df['good_quality_document']
    else:
        good_quality_condition = True

    tp_conditions = (expanded_df['cause'] == cause) & (expanded_df['ground_truth']) & (
        expanded_df['predicted_cause'])
    TP = len(expanded_df.loc[tp_conditions & good_quality_condition])

    fp_conditions = (expanded_df['cause'] == cause) & (~expanded_df['ground_truth']) & (
        expanded_df['predicted_cause'])
    FP = len(expanded_df.loc[fp_conditions & good_quality_condition])

    tn_conditions = (expanded_df['cause'] == cause) & (~expanded_df['ground_truth']) & (
        ~expanded_df['predicted_cause'])
    TN = len(expanded_df.loc[tn_conditions & good_quality_condition])

    fn_conditions = (expanded_df['cause'] == cause) & (expanded_df['ground_truth']) & (
        ~expanded_df['predicted_cause'])
    FN = len(expanded_df.loc[fn_conditions & good_quality_condition])

    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    precision = TP / (TP + FP) if (TP + FP) != 0 else np.nan

    return accuracy, recall, precision, (TP, FP, TN, FN)


def get_false_positives_negatives(expanded_df, cause):
    if cause != 'quality_is_not_ok':  # In this case, we remove the bad quality documents
        good_quality_condition = expanded_df['good_quality_document']
    else:
        good_quality_condition = True

    error_df = expanded_df.loc[
        (expanded_df['cause'] == cause) &
        ((expanded_df['ground_truth']) != (expanded_df['predicted_cause'])) &
        good_quality_condition].copy()
    error_df['error_type'] = error_df.apply(lambda x: 'FP' if x['predicted_cause'] else 'FN', axis=1)
    return error_df


def preprocess_file_name_to_extract_infos(file_name, all_documents):
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
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
            same_filename = invalid_restitutions_infos['formatted_filename'].apply(
                lambda x: x in normalize_str(file_name))
            same_plate_number = invalid_restitutions_infos['plateNumber'].apply(
                lambda x: x in file_name)
            all_documents[file_name]['cause'] = \
                invalid_restitutions_infos.loc[same_filename & same_plate_number,
                'adminComment'].values[0]

        all_documents[file_name]['plate_number'] = file_name.split('_')[0]
    else:
        file_path = ''
    return all_documents, file_path

def load_ground_truth_data(status):
    # Load ground truth and predictions
    ground_truth_path = f'data/performances_data/{status}_data/{status}_data_ground_truth.csv'  # Replace with your actual path
    ground_truth_data = pd.read_csv(ground_truth_path)
    return ground_truth_data

if __name__ == '__main__':

    RUN_ANALYSIS = True
    RUN_METRICS_COMPUTATION = True
    WITH_GPT = True
    PARTIAL_ANALYSIS = False # If true, you need to comment out irrelevant validation part in the ResultValidator class
    STATUS_TO_RUN = ['valid'#,
                     #'invalid'
                     ]
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    if RUN_ANALYSIS:
        invalid_restitutions_infos = pd.read_csv('data/links_to_dataset/invalid_restitutions.csv')
        invalid_restitutions_infos['formatted_filename'] = invalid_restitutions_infos['filename'].apply(
            lambda x: normalize_str(
                os.path.splitext(x.replace(' ', '_'))[0]))

        # Getting all the documents path and name

        all_documents = {}
        for status in STATUS_TO_RUN:
            image_directory = Path(f'data/performances_data/{status}_data/arval_classic_restitution_images/')
            image_files = os.listdir(image_directory)
            ground_truth_data = load_ground_truth_data(status)
            # Iterate over each image and perform the operations
            for file_name in image_files:
                try:

                    all_documents, file_path = preprocess_file_name_to_extract_infos(file_name, all_documents)

                except Exception as e:
                    print(e)
                    print(file_name)
                    print(file_path)
                    print(all_documents[file_name])
                    print("\n\n")

            # Analyze all documents and compare with the ground truth
            full_result_analysis = pd.DataFrame(
                columns=['document_name', 'true_status', 'predicted_status',
                         'true_cause', 'predicted_cause'])

            files_to_test = ground_truth_data['document_name'].values

            files_to_exclude = ['EH-082-TV_PV_de_restitution_.jpeg','FK-184-AJ_PV_de_restitution.png']
            files_to_iterate = {file: all_documents[file]
                                for file in sorted(files_to_test)
                                if file not in files_to_exclude}.items()
            i = 0
            print('file to iterate full list', files_to_iterate)
            print('lenght   =',len(files_to_iterate))
            for name, info in tqdm(files_to_iterate):
                logger.info(f"Analyzing {name}")
                try:
                    document_analyzer = ArvalClassicGPTDocumentAnalyzer(name, info['path'],
                                                                        #hyperparameters
                                                                        )
                    #document_analyzer.analyze()
                    if PARTIAL_ANALYSIS:
                        document_analyzer.assess_overall_quality()
                        document_analyzer.get_or_create_blocks()
                        document_analyzer.get_result_template()
                        try:
                            document_analyzer.analyze_block4_text(document_analyzer.block_4_info_path, # Change according to the analysis you want to perform
                                                                  verbose=False, plot_boxes=False)
                        except Exception as e:
                            document_analyzer.analyze_block4_text(document_analyzer.file_path_block4,
                                                                  verbose=False, plot_boxes=False)
                    else:
                        document_analyzer.analyze()
                    # document_analyzer.plot_blocks()
                    logger.info(f"Result: {document_analyzer.results}")

                    result_validator = ResultValidator(document_analyzer.results,
                                                       plate_number=info['plate_number'])
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
                    i += 1
                except Exception as e:
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
        saving_path = f'results/full_result_analysis_{status}.csv'
        full_result_analysis.to_csv(saving_path, index=False)


    print('Number of file analyse  =',i)

    if RUN_METRICS_COMPUTATION:

        for status in STATUS_TO_RUN:
            #predictions_path = 'results/predictions_amiel.csv'    # Replace with your actual path
            if RUN_ANALYSIS:
                predictions_path = saving_path
            else:
                predictions_path = 'results/full_result_analysis_20231222_133655.csv'

            predictions_data = pd.read_csv(predictions_path)
            merged_data = pd.merge(ground_truth_data, predictions_data, on='document_name', how='inner')
            os.makedirs(f'results/error_analysis_{status}_{dt}', exist_ok=True)
            merged_data.to_csv(f'results/error_analysis_{status}_{dt}/merged_data.csv', index=False)

            df = merged_data.copy()
            df['ground_truth'] = df['ground_truth'].apply(lambda x: x.replace('\n', '').replace(' ', '').split(','))  # Split string into list
            df.loc[df['predicted_cause'].isnull(), 'predicted_cause'] = '-'  # Replace NaN with '-'
            df['predicted_cause'] = df['predicted_cause'].apply(lambda x: x.replace('\n', '').replace(' ', '').split(','))  # Split string into list

            # List of all unique failure causes
            all_causes = set(
                cause for sublist in df.ground_truth.tolist() + df.predicted_cause.tolist() for cause in sublist)
            all_causes = ['quality_is_not_ok', 'signatures_are_not_ok', 'stamps_are_not_ok', 'mileage_is_not_ok',
                          'number_plate_is_not_filled', 'number_plate_is_not_right', 'block4_is_not_filled',
                          'block4_is_not_filled_by_company', 'driver_name_is_not_filled', 'serial_number_is_not_filled',
                          'telephone_is_not_filled', 'email_is_not_filled', 'block2_is_not_filled',
                          'restitution_date_is_not_filled']
            expanded_df = expand_df(df, all_causes)

            # Tag documents with good quality
            good_quality_documents = expanded_df.loc[
                (expanded_df['cause'] == 'quality_is_not_ok') &
                (~expanded_df['ground_truth'])].document_name.tolist()
            logger.info(f"Found {len(good_quality_documents)} good quality documents")

            expanded_df['good_quality_document'] = expanded_df['document_name'].isin(good_quality_documents)
            #all_causes = ['block4_is_not_filled', 'block4_is_not_filled_by_company',
            #              'driver_name_is_not_filled', 'email_is_not_filled', 'telephone_is_not_filled', 'quality_is_not_ok',
            #              ]

            summary_df = pd.DataFrame()
            for cause in sorted(all_causes):
                accuracy, recall, precision, (TP, FP, TN, FN) = calculate_metrics(expanded_df, cause)
                error_df = get_false_positives_negatives(expanded_df, cause)
                print(f"Error: {cause}")
                print(f" Accuracy: {accuracy:0.02f}")
                print(f" Recall: {recall:0.02f}")
                print(f" Precision: {precision:0.02f}")
                print(f" True Positives: {TP}")
                print(f" False Positives: {FP}")
                print(f" True Negatives: {TN}")
                print(f" False Negatives: {FN}\n")
                summary_df = pd.concat([summary_df,
                                        pd.DataFrame({
                                            'cause': [cause],
                                            'accuracy': [accuracy],
                                            'recall': [recall],
                                            'precision': [precision],
                                            'TP': [TP],
                                            'FP': [FP],
                                            'TN': [TN],
                                            'FN': [FN]
                                        }, index=[0]
                                        )])


                error_df.to_csv(f'results/error_analysis_{status}_{dt}/error_{cause}.csv', index=False)
            summary_df.to_csv(f'results/error_analysis_{status}_{dt}/summary.csv', index=False)