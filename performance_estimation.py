import numpy as np
from Levenshtein import distance

def compute_filling_accuracy(predicted, actual):
    total_keys = len(actual)
    correct_keys = sum([
        (predicted[key] == "<EMPTY>" and actual[key] is None) or
        (predicted[key] != "<NOTFOUND>" and predicted[key] == actual[key])
        for key in actual])
    return correct_keys / total_keys

def compute_content_accuracy(predicted, actual):
    total_keys = len(actual)
    correct_values = sum([predicted[key]['next'][1] == actual[key]['next'][1] for key in actual if 'next' in predicted[key] and 'next' in actual[key]])
    return correct_values / total_keys

def compute_content_fuzzy_accuracy(predicted, actual):
    lev_distances = [distance(str(predicted[key]['next'][1]), str(actual[key]['next'][1]))
                     for key in actual if 'next' in predicted[key] and 'next' in actual[key]]
    avg_lev_distance = np.mean(lev_distances) if lev_distances else 0
    max_possible_distance = max([max(len(str(predicted[key]['next'][1])), len(str(actual[key]['next'][1])))
                                 for key in actual if 'next' in predicted[key] and 'next' in actual[key]],
                                default=1)
    return 1 - (avg_lev_distance / max_possible_distance)

def compute_metrics_for_multiple_jsons(predicted_list, actual_list):
    all_metrics = {
        'general': {
            'Filling Accuracy': [],
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
                'Content Accuracy': [],
                'Content Fuzzy Accuracy': []
            }

        for block_name, block_values in actual.items():
            if block_name != 'File Name':
                for key, value in block_values.items():
                    key_predicted = {key: predicted[block_name][key]} if key in predicted[block_name] else {key: None}
                    key_actual = {key: value}

                    all_metrics['general']['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['general']['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['general']['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    all_metrics['by_file'][file_name]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_file'][file_name]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['by_file'][file_name]['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    # Update block-specific metrics
                    if block_name not in all_metrics['by_block']:
                        all_metrics['by_block'][block_name] = {
                            'Filling Accuracy': [],
                            'Content Accuracy': [],
                            'Content Fuzzy Accuracy': []
                        }

                    all_metrics['by_block'][block_name]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
                    all_metrics['by_block'][block_name]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
                    all_metrics['by_block'][block_name]['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(key_predicted, key_actual))

                    # Update key-specific metrics
                    if key not in all_metrics['by_key']:
                        all_metrics['by_key'][key] = {
                            'Filling Accuracy': [],
                            'Content Accuracy': [],
                            'Content Fuzzy Accuracy': []
                        }

                    all_metrics['by_key'][key]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
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

