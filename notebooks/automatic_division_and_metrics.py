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

# # Main functions

# +
import os 
DIR_CHANGED = False

if not DIR_CHANGED: 
    os.chdir('..')
# -

# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

from ai_documents.plotting import plot_boxes,plot_boxes_and_lines
from ai_documents.detection.pre_ocr_division import find_max_spacing_non_crossing_lines, cut_and_save_image, subdivide_batch_of_image
# -

# ## I. Automatic division single image

# +
from pipeline import get_processed_boxes_and_words_unguided_bloc

img_path = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/DM-984-VT_Proces verbal de restitution_page-0001.jpg"

hyperparameters = {'det_arch':"db_resnet50",
        'reco_arch': "crnn_mobilenet_v3_large",
        'pretrained': True ,
        'distance_margin': 10, # find_next_right_word for words_similarity
        'max_distance':  100, # find_next_right_word
        'minimum_overlap': 20 # find_next_right_word for _has_overlap
}


print(hyperparameters)

converted_boxes,image_dims = get_processed_boxes_and_words_unguided_bloc(img_path=img_path,
                                                            det_arch=hyperparameters['det_arch'],
                                                            reco_arch=hyperparameters['reco_arch'],
                                                            pretrained=hyperparameters['pretrained'],
                                                            verbose=False)
# -


plot_boxes(converted_boxes, image_dims)

img_height = image_dims[1]
line_num = 4
non_crossing_lines= find_max_spacing_non_crossing_lines(converted_boxes, img_height,line_num)
print("Non-crossing lines:", non_crossing_lines)

# +
plot_boxes_and_lines(converted_boxes, image_dims,non_crossing_lines)
output_fold = "data/performances_data/fleet_services_images/DM-984-VT_Proces_verbal_de_restitution_page-0001/autotest"
#output_fold= "data/performances_data/arval_classic_restitution_images/autotest"

cut_and_save_image(img_path, output_fold, non_crossing_lines)
# -

# ## II. Automatic division batch of image

# +

FOLDER_IMAGES = Path('data/performances_data/fleet_services_images')
subdivide_batch_of_image(FOLDER_IMAGES,4,hyperparameters,False)
    

# +
## III. Performance estimation 

# +
from performance_estimation import run_batch_analysis_undefined_blocs
from ai_documents.utils import clean_listdir

FOLDER_IMAGES = Path('data/performances_data/fleet_services_images')
image_list = clean_listdir(FOLDER_IMAGES, only="dir") 
print(FOLDER_IMAGES)
print(image_list)
all_res = run_batch_analysis_undefined_blocs(image_list,hyperparameters,verbose= False,plot_boxes=False)


# -

print(all_res)

# +
from ai_documents.utils import read_json
#all_res
# Define hyperparameter space
FOLDER_GROUND_TRUTHS = Path('data/performances_data/fleet_services_jsons')

# List of data
#image_list = clean_listdir(FOLDER_IMAGES, only="dir")
ground_truths_list = [x + '.json' for x in image_list]
actual_json_list = [read_json(FOLDER_GROUND_TRUTHS/filename) for filename in clean_listdir(FOLDER_GROUND_TRUTHS)]
print(actual_json_list)

# +
from performance_estimation import clean_predicted_data,compute_metrics_for_multiple_jsons
predicted_dict_list = [clean_predicted_data(results) for results in all_res]
print(predicted_dict_list)

compute_metrics_for_multiple_jsons(predicted_dict_list, actual_json_list)
