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
invalid_restitutions_infos['formatted_filename'] = invalid_restitutions_infos['filename'].apply(lambda x: normalize_str(
    os.path.splitext(x.replace(' ', '_'))[0]))

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

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0

    return accuracy, recall, precision, (TP, FP, TN, FN)


def get_false_positives_negatives(expanded_df, cause):
    error_df = expanded_df[
        (expanded_df['cause'] == cause) & ((expanded_df['ground_truth']) != (expanded_df['predicted_cause']))].copy()
    error_df['error_type'] = error_df.apply(lambda x: 'FP' if x['predicted_cause'] else 'FN', axis=1)
    return error_df

if __name__ == '__main__':

    RUN_ANALYSIS = False
    RUN_METRICS_COMPUTATION = True
    WITH_GPT = True

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load ground truth and predictions
    ground_truth_path = 'results/invalid_data_ground_truth.csv'  # Replace with your actual path
    ground_truth_data = pd.read_csv(ground_truth_path)

    if RUN_ANALYSIS:

        # Analyze all documents and compare with the ground truth
        full_result_analysis = pd.DataFrame(
            columns=['document_name', 'true_status', 'predicted_status', 'true_cause', 'predicted_cause'])
        # for name, info in all_documents.items():

        files_to_test = ground_truth_data['document_name'].values
        #files_to_test = ["ES-337-RE_PVR.jpeg"]
        files_to_exclude = ['EM-272-VS_Document_p1.jpeg', # Thumb disturbance
                            'EZ-912-QS_PV_de_reprise_p2.jpeg',# Content policy violation
                            "FA-256-WW_PV_de_Restitution_p1.jpeg", # Content policy violation
                            "FF-121-EK_PV_p1.jpeg", # content_policy_violation
                            "FF-173-LL_PV_restitution.jpeg", # content_policy_violation
                            "FF-403-FX_PV_de_reprise_p1.jpeg",# content_policy_violation
                            "FF-724-NB_pV_restitution__p1.jpeg", # content_policy_violation
                            "FK-468-LV_PV_de_reprise_p1.jpeg", # content_policy_violation
                            "FL-354-QG_PV_de_reprise_p1.jpeg", # content_policy_violation
                        ]

        files_to_iterate = {file: all_documents[file]
                            for file in sorted(files_to_test)
                            if file not in files_to_exclude}.items()

        for name, info in tqdm(files_to_iterate):

            try:
                document_analyzer = ArvalClassicGPTDocumentAnalyzer(name, info['path'],
                                                                    #hyperparameters
                                                                    )
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
        saving_path = f'results/full_result_analysis_{dt}.csv'
        full_result_analysis.to_csv(saving_path, index=False)

    if RUN_METRICS_COMPUTATION:

        #predictions_path = 'results/predictions_amiel.csv'    # Replace with your actual path
        predictions_path = 'results/full_result_analysis_20231221_175556.csv'
        #predictions_path = saving_path
        predictions_data = pd.read_csv(predictions_path)
        merged_data = pd.merge(ground_truth_data, predictions_data, on='document_name', how='inner')
        os.makedirs(f'results/error_analysis_{dt}', exist_ok=True)

        merged_data.to_csv(f'results/error_analysis_{dt}/merged_data.csv', index=False)
        """
        ground_truth_data = read_csv(ground_truth_path) #TODO: rewrite using dataframes
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

        pd.DataFrame(tp_fp_tn_fn_per_error).T.to_csv(f'results/metrics_per_error_{dt}.csv')
        """

        df = merged_data.copy()
        df['ground_truth'] = df['ground_truth'].apply(lambda x: x.replace('\n', '').replace(' ', '').split(','))  # Split string into list
        df['predicted_cause'] = df['predicted_cause'].apply(lambda x: x.replace('\n', '').replace(' ', '').split(','))  # Split string into list

        # List of all unique failure causes
        all_causes = set(
            cause for sublist in df.ground_truth.tolist() + df.predicted_cause.tolist() for cause in sublist)

        expanded_df = expand_df(df, all_causes)

        # Tag documents with bad quality
        good_quality_documents = expanded_df.loc[
            (expanded_df['cause'] != 'quality_is_not_ok') &
            (expanded_df['ground_truth'])].document_name.tolist()

        expanded_df['good_quality_document'] = expanded_df['document_name'].isin(good_quality_documents)

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
            error_df.to_csv(f'results/error_analysis_{dt}/error_{cause}.csv', index=False)
