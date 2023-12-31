import json
import os
import unicodedata

normalize_str = lambda s: ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

def clean_listdir(path_to_folder,only="file"):
    if only is None:
        return [listed for listed in os.listdir(path_to_folder) if not listed.startswith('.')]
    elif only == "file":
        return [listed for listed in os.listdir(path_to_folder) if not listed.startswith('.') and os.path.isfile(path_to_folder/listed)]
    elif only == "dir":
        return [listed for listed in os.listdir(path_to_folder) if not listed.startswith('.')  and os.path.isdir(path_to_folder/listed)]


def read_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data

def get_result_template(folder_ground_truths):
    json_template = {}
    ground_truths_list = [x for x in clean_listdir(folder_ground_truths)]
    sample_json = read_json(folder_ground_truths / f'{ground_truths_list[0]}')
    block_names = [k for k in sample_json.keys() if k.startswith('block')]
    for bn in block_names:
        json_template[bn] = sample_json[bn].keys()
    return json_template

def has_non_none_attributes(obj, *attrs):
    """
    Check if an object has non-None attributes
    :param obj:
    :param attrs:
    :return:
    """
    return all(hasattr(obj, attr) and getattr(obj, attr) is not None for attr in attrs)
