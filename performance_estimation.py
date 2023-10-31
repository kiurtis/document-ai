import numpy as np
from Levenshtein import distance


def compute_filling_accuracy(predicted, actual):
    total_keys = len(actual)
    correct_keys = sum([(predicted[key] is None and actual[key] is None) or
                        (predicted[key] is not None and actual[key] is not None)
                        for key in actual])
    return correct_keys / total_keys


def compute_content_accuracy(predicted, actual):
    total_keys = len(actual)
    correct_values = sum([predicted[key] == actual[key] for key in actual])
    return correct_values / total_keys


def compute_content_fuzzy_accuracy(predicted, actual):
    lev_distances = [distance(str(predicted[key]), str(actual[key]))
                     for key in actual if predicted[key] is not None and actual[key] is not None]
    avg_lev_distance = np.mean(lev_distances) if lev_distances else 0
    max_possible_distance = max([max(len(str(predicted[key])), len(str(actual[key])))
                                 for key in actual if predicted[key] is not None and actual[key] is not None],
                                default=1)
    return 1 - (avg_lev_distance / max_possible_distance)


def compute_metrics_for_multiple_jsons(predicted_list, actual_list):
    all_metrics = {
        'general': {
            'Filling Accuracy': [],
            'Content Accuracy': [],
            'Content Fuzzy Accuracy': []
        },
        'by_key': {}
    }

    for predicted, actual in zip(predicted_list, actual_list):
        all_metrics['general']['Filling Accuracy'].append(compute_filling_accuracy(predicted, actual))
        all_metrics['general']['Content Accuracy'].append(compute_content_accuracy(predicted, actual))
        all_metrics['general']['Content Fuzzy Accuracy'].append(compute_content_fuzzy_accuracy(predicted, actual))

        for key in actual:
            if key not in all_metrics['by_key']:
                all_metrics['by_key'][key] = {'Filling Accuracy': [], 'Content Accuracy': [],
                                              'Content Fuzzy Accuracy': []}

            # Key-specific metrics
            key_predicted = {key: predicted[key]} if key in predicted else {key: None}
            key_actual = {key: actual[key]}

            all_metrics['by_key'][key]['Filling Accuracy'].append(compute_filling_accuracy(key_predicted, key_actual))
            all_metrics['by_key'][key]['Content Accuracy'].append(compute_content_accuracy(key_predicted, key_actual))
            all_metrics['by_key'][key]['Content Fuzzy Accuracy'].append(
                compute_content_fuzzy_accuracy(key_predicted, key_actual))

    # Compute averages for metrics
    for metric in all_metrics['general']:
        all_metrics['general'][metric] = np.mean(all_metrics['general'][metric])

    for key in all_metrics['by_key']:
        for metric in all_metrics['by_key'][key]:
            all_metrics['by_key'][key][metric] = np.mean(all_metrics['by_key'][key][metric])

    return all_metrics