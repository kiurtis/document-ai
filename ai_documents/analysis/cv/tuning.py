import json
from pathlib import Path
import random
from datetime import datetime
from collections import defaultdict

import numpy as np
from Levenshtein import distance
from loguru import logger

from ai_documents.utils import clean_listdir, read_json, get_result_template
from ai_documents.analysis.cv.boxes_processing import get_processed_boxes_and_words, postprocess_boxes_and_words, \
    postprocess_boxes_and_words_unguided_bloc
from ai_documents.plotting import plot_boxes_with_text
from document_parsing import find_next_right_word

FOLDER_GROUND_TRUTHS = Path('../../../data/performances_data/valid_data/fleet_services_jsons')
FOLDER_IMAGES = Path('../../../data/performances_data/valid_data/fleet_services_images')
RESULT_TEMPLATE = get_result_template(
    folder_ground_truths=FOLDER_GROUND_TRUTHS,
)


def compute_filling_accuracy(predicted, actual):
    total_keys = len(actual)
    correct_keys = sum([
        (predicted[key] == "<EMPTY>" and actual[key] is None) or
        (predicted[key] == None and actual[key] is None) or
       #(predicted[key] == "<NOT_FOUND>" and actual[key] is None) or
        ((predicted[key] != "<NOT_FOUND>" and predicted[key] != "<EMPTY>") and actual[key] is not None )
        for key in actual])
    return correct_keys / total_keys

def compute_false_positive_filling_accuracy(predicted, actual):
    total_keys = len(actual)
    false_positives = sum([
        (predicted[key] != "<NOT_FOUND>" and predicted[key] != "<EMPTY>" and actual[key] is None)
        for key in actual
    ])
    # Avoid division by zero
    if total_keys == 0:
        return None  # or 0, depending on how you want to handle this edge case

    return false_positives / total_keys



def compute_content_accuracy(predicted, actual):
    total_keys = len(actual)
    
    #correct_values = sum([predicted[key]['next'][1] == actual[key]['next'][1] for key in actual if 'next' in predicted[key] and 'next' in actual[key]])
    correct_values = sum([
        (predicted[key] == actual[key]) or
        (predicted[key] == "<NOT_FOUND>" and actual[key] == None) or
        (predicted[key] == "<EMPTY>" and actual[key] == None ) or
        (predicted[key] == None and actual[key] == None ) 
        for key in actual])
    
    return correct_values / total_keys

def extended_normalized_levenstein_distance(predicted, actual):
    if (predicted == "<EMPTY>" and actual is None) \
            or (predicted is None and actual is None):
        return 0 # Filling True negative: no distance
    elif (predicted == "<NOT_FOUND>") \
            or (predicted == "<EMPTY>" and actual is not None):
        return 1 # Filling False negative: max  distance
    elif (predicted not in ("<EMPTY>", "<NOT_FOUND>") and actual is None):
        return 1 # Filling False positive: max distance
    else:
        return min(distance(predicted, actual) / len(actual), 1) # Content distance: normalized Levenstein distance

def compute_content_fuzzy_accuracy(predicted, actual):
    lev_distances = [extended_normalized_levenstein_distance(str(predicted[key]), str(actual[key]))
                     for key in actual]

    avg_lev_distance = np.mean(lev_distances) if lev_distances else 0
    
    return 1 - avg_lev_distance


def compute_metrics_for_multiple_jsons(predicted_list, actual_list):
    logger.info('Computing metrics for multiple JSONs 2')
    all_metrics = {
        'general': {
            'Filling Accuracy': [],
            'FP Filling Accuracy': [],
            'Content Accuracy': [],
            'Content Fuzzy Accuracy': []
        },
        'by_file': {},
        'by_block': {},
        'by_key': {}
    }

    for predicted, actual in zip(predicted_list, actual_list):
        file_name = actual['File Name']

        if file_name not in all_metrics['by_file']:
            all_metrics['by_file'][file_name] = {
                'Filling Accuracy': [],
                'FP Filling Accuracy': [],
                'Content Accuracy': [],
                'Content Fuzzy Accuracy': []
            }

        for block_name, block_values in actual.items():
            if block_name != 'File Name':
                for key, value in block_values.items():
                    key_predicted = {key: predicted[block_name][key]} if key in predicted[block_name] else {key: None}
                    key_actual = {key: value}

                    all_metrics['general']['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['general']['FP Filling Accuracy'].append(compute_false_positive_filling_accuracy(key_predicted, key_actual))
                    all_metrics['general']['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['general']['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    all_metrics['by_file'][file_name]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_file'][file_name]['FP Filling Accuracy'].append(compute_false_positive_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_file'][file_name]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['by_file'][file_name]['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    # Update block-specific metrics
                    if block_name not in all_metrics['by_block']:
                        all_metrics['by_block'][block_name] = {
                            'Filling Accuracy': [],
                            'FP Filling Accuracy': [],
                            'Content Accuracy': [],
                            'Content Fuzzy Accuracy': []
                        }

                    all_metrics['by_block'][block_name]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_block'][block_name]['FP Filling Accuracy'].append(compute_false_positive_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_block'][block_name]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['by_block'][block_name]['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    # Update key-specific metrics
                    if key not in all_metrics['by_key']:
                        all_metrics['by_key'][key] = {
                            'Filling Accuracy': [],
                            'FP Filling Accuracy': [],
                            'Content Accuracy': [],
                            'Content Fuzzy Accuracy': []
                        }

                    all_metrics['by_key'][key]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_key'][key]['FP Filling Accuracy'].append(compute_false_positive_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_key'][key]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['by_key'][key]['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

    # Compute averages for metrics
    for metric in all_metrics['general']:
        all_metrics['general'][metric] = np.mean(all_metrics['general'][metric])

    for file_name in all_metrics['by_file']:
        for metric in all_metrics['by_file'][file_name]:
            all_metrics['by_file'][file_name][metric] = np.mean(all_metrics['by_file'][file_name][metric])

    for block_name in all_metrics['by_block']:
        for metric in all_metrics['by_block'][block_name]:
            all_metrics['by_block'][block_name][metric] = np.mean(all_metrics['by_block'][block_name][metric])

    for key in all_metrics['by_key']:
        for metric in all_metrics['by_key'][key]:
            all_metrics['by_key'][key][metric] = np.mean(all_metrics['by_key'][key][metric])

    return all_metrics




def has_found_box(value):
    return type(value) == tuple


def run_batch_analysis(image_list, hyperparameters, verbose, plot_boxes=False):
    all_results = []
    for element in image_list:
        print(f'==== Running for file: {element} =====')
        filename_prefix = f'{element[:-5]}'

        result_json = {}
        result_json['File Name'] = element
        for bn in list(RESULT_TEMPLATE.keys()):
            result_json[bn] = {}
            path_to_name_jpeg = FOLDER_IMAGES / element / 'blocks' / (element + f"_{bn.replace('_', ' ')}.png")
            converted_boxes = get_processed_boxes_and_words(img_path=path_to_name_jpeg,
                                                            block=bn,
                                                            det_arch=hyperparameters['det_arch'],
                                                            reco_arch=hyperparameters['reco_arch'],
                                                            pretrained=hyperparameters['pretrained'],
                                                            verbose=verbose)
            if plot_boxes:
                plot_boxes_with_text(converted_boxes)
            converted_boxes = postprocess_boxes_and_words(converted_boxes,
                                                          block=bn,
                                                          verbose=verbose,
                                                          safe=True)
            for key_word in RESULT_TEMPLATE[bn]:
                if verbose:
                    print(f'Running {key_word}')
                result_json[bn][key_word] = find_next_right_word(converted_boxes, key_word,
                                                                 distance_margin=hyperparameters['distance_margin'],
                                                                 max_distance=hyperparameters['max_distance'],
                                                                 minimum_overlap=hyperparameters['minimum_overlap'],
                                                                 verbose=verbose)
                if has_found_box(result_json[bn][key_word]):
                    result_json[bn][key_word] = result_json[bn][key_word]['next']
        all_results.append(result_json)
    return all_results



def run_batch_analysis_undefined_blocs(image_list, hyperparameters, verbose, plot_boxes=False):
    all_results = []
    print(RESULT_TEMPLATE)
    for element in image_list:
        print(f'==== Running for file: {element} =====')
        filename_prefix = f'{element[:-5]}'

        result_json = {}
        result_json['File Name'] = element

        folder_with_blocs = FOLDER_IMAGES/element/'automaticbloc'
        bloc_list = clean_listdir(folder_with_blocs)
        print(bloc_list)
        for bloc in bloc_list:
       
            path_to_name_jpeg = FOLDER_IMAGES/element/'automaticbloc'/bloc
            print(path_to_name_jpeg)

            converted_boxes = get_processed_boxes_and_words(img_path=path_to_name_jpeg,
                                                            block=1,
                                                            det_arch=hyperparameters['det_arch'],
                                                            reco_arch=hyperparameters['reco_arch'],
                                                            pretrained=hyperparameters['pretrained'],
                                                            verbose=verbose)
            if plot_boxes:
                plot_boxes_with_text(converted_boxes)
                
            converted_boxes = postprocess_boxes_and_words_unguided_bloc(converted_boxes,
                                                                      verbose=verbose,
                                                                      safe=True)

            for bn in list(RESULT_TEMPLATE.keys()):
                if bn not in result_json:
                    result_json[bn] = {}

                for key_word in RESULT_TEMPLATE[bn]:
                    if verbose:
                        print(f'Running {key_word}')
                    
                    if bn in result_json and key_word in result_json[bn]:
                        if result_json[bn][key_word] == "<EMPTY>" or result_json[bn][key_word] == "<NOT_FOUND>": 
                            result_json[bn][key_word] = find_next_right_word(converted_boxes, key_word,
                                                                     distance_margin=hyperparameters['distance_margin'],
                                                                     max_distance=hyperparameters['max_distance'],
                                                                     minimum_overlap=hyperparameters['minimum_overlap'],
                                                                     verbose=verbose)
                            if has_found_box(result_json[bn][key_word]):
                                result_json[bn][key_word] = result_json[bn][key_word]['next']
                    else:
                        result_json[bn][key_word] = find_next_right_word(converted_boxes, key_word,
                                                                     distance_margin=hyperparameters['distance_margin'],
                                                                     max_distance=hyperparameters['max_distance'],
                                                                     minimum_overlap=hyperparameters['minimum_overlap'],
                                                                     verbose=verbose)
                        if has_found_box(result_json[bn][key_word]):
                            result_json[bn][key_word] = result_json[bn][key_word]['next']
        all_results.append(result_json)
    return all_results





def clean_predicted_data(data):
    new_data = {}
    new_data['File Name'] = data['File Name']

    for block, values in [(el, _) for (el, _) in data.items() if el.startswith('block')]:
        new_data[block] = {}
        if isinstance(values, dict):
            for key, content in values.items():
                if isinstance(content, dict) and 'next' in content:
                    new_data[block][key] = content['next'][1]
                else:
                    new_data[block][key] = content
    return new_data

def sample_hyperparameters(hyperparameter_space):
    return {key: random.choice(value) for key, value in hyperparameter_space.items()}

# Function to execute a single iteration of hyperparameter optimization
def perform_optimization_iteration(image_list, hyperparameter_space, results_list_path):
    hyperparameters = sample_hyperparameters(hyperparameter_space)
    start = datetime.now()

    predicted_dict_list = run_batch_analysis(image_list, hyperparameters, verbose=False)
    predicted_dict_list = [clean_predicted_data(results) for results in predicted_dict_list]
    predicted_dict_list = sorted(predicted_dict_list, key=lambda x: x['File Name'])

    actual_json_list = [read_json(FOLDER_GROUND_TRUTHS / filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]
    actual_json_list = sorted(actual_json_list, key=lambda x: x['File Name'])

    metrics = compute_metrics_for_multiple_jsons(predicted_dict_list, actual_json_list)

    end = datetime.now()
    duration = end - start

    result = {
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'duration': str(duration)
    }

    # Append results to a JSON file for persistent storage
    with open(results_list_path, 'a') as f:
        f.write(json.dumps(result) + "\n")  # Appends the result as a new line in the file

    return result


def perform_hyperparameter_optimization(num_iterations, hyperparameter_space, results_list_path,
                                        folder_images, verbose):

    # List of data
    image_list = clean_listdir(folder_images, only="dir")

    # Ensure the results file is empty before starting
    open(results_list_path, 'w').close()

    # Perform optimization iterations
    for _ in range(num_iterations):
        result = perform_optimization_iteration(image_list, hyperparameter_space, results_list_path)
        if verbose:
            print(f"Iteration completed with result: {result}")

    # Read the results file line by line for analysis
    results_list = []
    with open(results_list_path, 'r') as f:
        for line in f:
            results_list.append(json.loads(line.strip()))

    # The results_list can now be used for further analysis

def parse_random_search_results(results_file_path):
    # Initialize a dictionary to keep the best results for the general section
    best_results_general = defaultdict(lambda: {'value': float('-inf'), 'hyperparameters': None})

    # Dictionaries to keep the best results for each file, block, and key
    best_results_by_file = defaultdict(lambda: defaultdict(lambda: {'value': float('-inf'), 'hyperparameters': None}))
    best_results_by_block = defaultdict(lambda: defaultdict(lambda: {'value': float('-inf'), 'hyperparameters': None}))
    best_results_by_key = defaultdict(lambda: defaultdict(lambda: {'value': float('-inf'), 'hyperparameters': None}))

    # Read the results from the JSON file
    with open(results_file_path, 'r') as file:
        for line in file:
            result = json.loads(line.strip())
            hyperparameters = result['hyperparameters']
            metrics = result['metrics']

            # Update the best results for the general metrics
            for metric, value in metrics['general'].items():
                if value > best_results_general[metric]['value']:
                    best_results_general[metric] = {
                        'value': value,
                        'hyperparameters': hyperparameters
                    }

            # Function to update the best results for 'by_file', 'by_block', or 'by_key'
            def update_best_results(best_results_section, metrics_section):
                for identifier, metrics_data in metrics_section.items():
                    for metric, value in metrics_data.items():
                        if value > best_results_section[identifier][metric]['value']:
                            best_results_section[identifier][metric] = {
                                'value': value,
                                'hyperparameters': hyperparameters
                            }

            # Update the best results for 'by_file', 'by_block', and 'by_key'
            update_best_results(best_results_by_file, metrics['by_file'])
            update_best_results(best_results_by_block, metrics['by_block'])
            update_best_results(best_results_by_key, metrics['by_key'])

    # Convert defaultdicts to regular dicts for JSON serialization
    best_results_general = dict(best_results_general)
    best_results_by_file = {file: dict(metrics) for file, metrics in best_results_by_file.items()}
    best_results_by_block = {block: dict(metrics) for block, metrics in best_results_by_block.items()}
    best_results_by_key = {key: dict(metrics) for key, metrics in best_results_by_key.items()}

    return {
        'general': best_results_general,
        'by_file': best_results_by_file,
        'by_block': best_results_by_block,
        'by_key': best_results_by_key
    }


if __name__ == "__main__":
    # Define hyperparameter space

    VERBOSE = True
    # List of data
    image_list = clean_listdir(FOLDER_IMAGES, only="dir")
    ground_truths_list = [x + '.json' for x in image_list]

    # Define number of iterations and results file path
    NUM_ITERATIONS = 10
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_LIST_PATH = f'hyperparameters_optimization_results_{dt}.json'
    DET_ARCHS = [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
    ]
    DET_ROT_ARCHS = ["db_resnet50_rotation"]

    RECO_ARCHS = [
        "crnn_vgg16_bn",
        "crnn_mobilenet_v3_small",
        "crnn_mobilenet_v3_large",
        "sar_resnet31",
        "master",
        "vitstr_small",
        "vitstr_base",
        "parseq",
    ]

    HYPERPARAMETER_SPACE = {'det_arch': DET_ARCHS + DET_ROT_ARCHS,
                            'reco_arch': RECO_ARCHS,
                            'pretrained': [True, ],
                            'distance_margin': [1, 2, 5, 10, 20],  # find_next_right_word for words_similarity
                            'max_distance': [10, 50, 100, 200, 500],  # find_next_right_word
                            'minimum_overlap': [1, 2, 5, 10, 20, 50, 100]  # find_next_right_word for _has_overlap
                            }

    hyperparameters = {k: v[0] for (k, v) in HYPERPARAMETER_SPACE.items()}

    perform_hyperparameter_optimization(num_iterations=NUM_ITERATIONS,
                                        hyperparameter_space=HYPERPARAMETER_SPACE,
                                        folder_images=FOLDER_IMAGES,
                                        results_list_path=RESULTS_LIST_PATH,
                                        verbose=VERBOSE)
