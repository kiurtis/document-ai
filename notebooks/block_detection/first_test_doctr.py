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

# + jupyter={"outputs_hidden": false} pycharm={"is_executing": true}
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from Levenshtein import distance
from shapely import Polygon

from pprint import pprint 


# +
def get_attribute_size(attribute):
    return len(attribute.split(' '))
    
def retrieve_attribute(block_idx, line_idx, loc_idx, s, attribute_to_retrieve):
    block = result.export()['pages'][0]['blocks'][block_idx]
    words =  block['lines'][line_idx]['words']
    try:
        attribute = words[loc_idx + s + 1]
    except:
        attribute = {'value': 'EMPTY'}
    return attribute

def get_bl_coordinates_containing_attribute(model_result, attribute, page=0, distance_margin=0):
    attribute_size = get_attribute_size(attribute)
    coordinates_list = []
    for b_idx, b in enumerate(model_result.export()['pages'][page]['blocks']):
         for l_idx, l in enumerate(b['lines']): # line
            
            for loc_idx, _ in enumerate(l['words']): # word location in the line
                candidate_attribute = b['lines'][l_idx]['words'][loc_idx]['value']
                max_size_from_loc = len(l['words']) - loc_idx - 1  # Remaining words in the line
                max_extension = min(attribute_size,max_size_from_loc)
                
                coordinates_list.append((b_idx, l_idx, loc_idx, 0, candidate_attribute, distance(candidate_attribute, attribute))) 

                for extension in range(0,max_extension):
                    candidate_attribute = b['lines'][l_idx]['words'][loc_idx]['value']
                    for s in range(0, extension): # We test all the group of words smaller than attribute
                        #print('bla',s,candidate_attribute)
                        candidate_attribute += ' ' + b['lines'][l_idx]['words'][loc_idx+s+1]['value']
                        coordinates_list.append((b_idx, l_idx, loc_idx, s+1, candidate_attribute, distance(candidate_attribute, attribute))) 
            
    selected_coordinates_list = [coords for coords in coordinates_list if coords[5] <= distance_margin]
    selected_coordinates_list = sorted(selected_coordinates_list, key=lambda vals:vals[5], reverse=False)

    selected_coordinates_list = [tup for tup in  selected_coordinates_list]
    print(selected_coordinates_list)
    return {"block_idx":selected_coordinates_list[0][0],
           "line_idx":selected_coordinates_list[0][1],
           "loc_idx":selected_coordinates_list[0][2],
           "s":selected_coordinates_list[0][3],
           "attribute_to_retrieve":selected_coordinates_list[0][4],
           "distance":selected_coordinates_list[0][5]} 

def get_value(model_result, attribute, distance_margin=0):
    closest_key = get_bl_coordinates_containing_attribute(model_result, attribute, page=0, distance_margin=distance_margin)
    closest_key.pop('distance')
    res = retrieve_attribute(**closest_key)
    return res 


# -

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_pdf("../arval/pvl_GM266PC.pdf")
result = model(doc)

# + jupyter={"outputs_hidden": true}
result.export()
# -

for attribute in ('Prénom', 'Nom', 'Mail', 'Mobile', 'Marque','Kilomètres au compteur'):
    value = get_value(result, attribute,distance_margin=1)
    print(f'Value for "{attribute}": {value["value"]}')
    print('\n')

attribute = "Immatriculation (ou numéro de série)"
value = get_value(result, attribute, distance_margin=12)
value

# +
# Doc 2

# PDF
doc = DocumentFile.from_pdf("../arval/pvl_GN074AF.pdf")
result = model(doc)

# -

for attribute in ('Prénom', 'Nom', 'Mail', 'Mobile', 'Marque','Kilomètres au compteur'):
    try:
        value = get_value(result, attribute,distance_margin=1)
        print(f'Value for "{attribute}": {value["value"]}')
        print('\n')
    except:
        pass

# +

import matplotlib.pyplot as plt
plt.imshow(doc[0])
# -

result = model([doc[0]])

result.show([doc[0]])

result.export()['pages'][0]['blocks'][12]

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_pdf("/Users/amielsitruk/Downloads/16265164-e5a2-11ed-908f-9f6f44c54eb9-eb41fa9797-assolement_scea bonnassieux_83393644600016_2022.pdf")
#result = model(doc)
#synthetic_pages = result.synthesize()


result.show(doc)



