import math
from pdf2image import convert_from_path
from Levenshtein import distance as l_distance

def convert_pdf_to_jpg(pdf_path, output_folder=None, dpi=300):
    # Convert PDF to list of images
    images = convert_from_path(pdf_path, dpi)

    # Save images to the output directory
    for i, image in enumerate(images):
        image_name = f"output_page_{i + 1}.jpg"
        if output_folder:
            image_name = f"{output_folder}/{image_name}"
        image.save(image_name, "JPEG")

def convert_to_cartesian(bound, img_dims):
    '''Convert data from matrix-like coordinates system (x,y), with y from top to bottom to a
    cartesian coordinates system (x,y) with y from bottom to top.
    Return coordinates with the format x_min, x_max, y_min, y_max'''
    _, img_height = img_dims
    cartesian_bound = []

    for box, word in bound:
        # Convert y-values by subtracting them from the image height
        x_min, x_max, y_min, y_max = box[0], box[1], img_height - box[3], img_height - box[2]
        new_box = [x_min, x_max, y_min, y_max]
        cartesian_bound.append((new_box, word))

    return cartesian_bound

def convert_coordinates(geometry, page_dim):
    len_x = page_dim[1]
    len_y = page_dim[0]
    (x_min, y_min) = geometry[0]
    (x_max, y_max) = geometry[1]
    x_min = math.floor(x_min * len_x)
    x_max = math.ceil(x_max * len_x)
    y_min = math.floor(y_min * len_y)
    y_max = math.ceil(y_max * len_y)
    return [x_min, x_max, y_min, y_max]


def get_block_coordinates(output):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    i = 0
    for obj1 in output['pages'][0]["blocks"]:
        converted_coordinates = convert_coordinates(
            obj1["geometry"], page_dim
        )
        print("{}: {}".format(converted_coordinates,
                              i
                              )
              )
        text_coordinates.append(converted_coordinates)
        i += 1
    return text_coordinates


def get_words_coordinates(output, verbose):
    page_dim = output['pages'][0]["dimensions"]
    text_coordinates = []
    text_coordinates_and_word = []
    for obj1 in output['pages'][0]["blocks"]:
        for obj2 in obj1["lines"]:
            for obj3 in obj2["words"]:
                converted_coordinates = convert_coordinates(obj3["geometry"], page_dim)
                if verbose:
                    print("{}: {}".format(converted_coordinates, obj3["value"]))
                text_coordinates.append(converted_coordinates)
                text_coordinates_and_word.append((converted_coordinates, obj3["value"]))
    return text_coordinates, text_coordinates_and_word


def _compare_words_similarity(word1, word2, distance_margin):
    return l_distance(word1, word2) <= distance_margin


def _has_overlap(y1_min, y1_max, y2_min, y2_max, minimum_overlap=10):
    overlap_start = max(y1_min, y2_min)
    overlap_end = min(y1_max, y2_max)

    # If the segments overlap, return the overlap length, otherwise return 0
    return max(0, overlap_end - overlap_start - minimum_overlap)


def get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin, verbose):
    # Find the bounding box for the search word
    key_box = None
    for box, word in bounding_boxes:
        if _compare_words_similarity(word.lower(), key_word.lower(), distance_margin):
            if verbose:
                print(word, key_word)
            key_box = box
            break
    if verbose:
        print(f'Key box defined to {key_box}')

    return key_box, key_word


def compute_box_distance(box_1, box_2, mode="euclidian"):
    """
    Compute the Euclidean distance between the centers of two boxes.

    Parameters:
    - key_box (tuple): A tuple representing the first box (key_x_min, key_x_max, key_y_min, key_y_max).
    - box (tuple): A tuple representing the second box (b_x_min, b_x_max, b_y_min, b_y_max).

    Returns:
    - float: The Euclidean distance between the centers of the two boxes.
    """
    x_min_1, x_max_1, y_min_1, y_max_1 = box_1
    x_min_2, x_max_2, y_min_2, y_max_2 = box_2

    if mode == "euclidian":

        # Calculate the center coordinates of each box
        center_x_1 = (x_min_1 + x_max_1) / 2
        center_y_1 = (y_min_1 + y_max_1) / 2
        center_x_2 = (x_min_2 + x_max_2) / 2
        center_y_2 = (y_min_2 + y_max_2) / 2

        # Compute the Euclidean distance between the centers
        distance = ((center_x_1 - center_x_2) ** 2 + (center_y_1 - center_y_2) ** 2) ** 0.5
    elif mode == "right_horizontal":
        x_max_2 - x_max_1  # Calculate distance right coordinates

    return distance


def find_next_right_word(bounding_boxes, key_word, distance_margin=2, max_distance=200, minimum_overlap=10,
                         box_distance_mode='euclidian', verbose=False):
    """
    Find the closest word to the right of a given key word based on bounding boxes.

    Parameters:
    - bounding_boxes (list): A list of tuples where each tuple contains a bounding box and a word.
                             Each bounding box is represented as (x_min, x_max, y_min, y_max).
    - key_word (str): The reference word to which the closest right-side word is to be found.
    - distance_margin (int, optional): Margin to consider while matching the key word with words in bounding_boxes.
                                       Defaults to 1.
    - max_distance (int, optional): Maximum allowed distance between key_word and the word to its right.
                                    Defaults to 200.
    - verbose (bool, optional): If True, the function will print additional debug information. Defaults to False.

    Returns:
    - dict: A dictionary containing the bounding box and word for both the detected key word
            and the closest word to its right. If no word is found to the right, the 'next' key will have None values.

    Example:
    ```
    bounding_boxes = [((10, 20, 10, 20), "hello"), ((25, 35, 10, 20), "world")]
    find_next_right_word(bounding_boxes, "hello")
    # Output: {'detected': ((10, 20, 10, 20), 'hello'), 'next': ((25, 35, 10, 20), 'world')}
    ```

    Note:
    The function considers words to be on the same line if their y-coordinates overlap.
    """
    key_box, key_word = get_box_corresponding_to_word(key_word, bounding_boxes, distance_margin, verbose)

    # If the word isn't found, return None
    if key_box is None:
        if verbose:
            print("No box corresponding to the key word found")
        return "<NOT_FOUND>"

    # Variables to store the closest word and its distance
    found_value = False
    closest_distance = float('inf')
    key_x_min, key_x_max, key_y_min, key_y_max = key_box
    for box, word in bounding_boxes:
        # Check if the word's y-coordinates overlap with the key word's
        b_x_min, b_x_max, b_y_min, b_y_max = box
        overlap = _has_overlap(key_y_min, key_y_max, b_y_min, b_y_max, minimum_overlap=minimum_overlap)
        if overlap:
            if verbose:
                print(f'{word} overlaps {key_word}')
            # Check if the word is to the right of the key word
            if b_x_min > key_x_min:  # There maybe x overlap as well

                distance = compute_box_distance(key_box,
                                                box,
                                                mode=box_distance_mode)
                if verbose:
                    print(f'distance to {word}', distance)
                if distance < min(closest_distance, max_distance):
                    closest_distance = distance
                    closest_word = word
                    closest_box = box
                    found_value = True

    if not found_value:
        return "<EMPTY>"

    return {'detected': (key_box, key_word),
            'next': (closest_box, closest_word)}


def merge_word_with_following(converted_boxes, key_word, distance_margin=1, max_distance=200, verbose=False,
                              safe=False):
    """
    Merge the bounding box and word of a given key_word with its closest following word.

    Parameters:
    - converted_boxes (list of tuple): A list of tuples, where each tuple contains a bounding box and a word.
                                      Each bounding box is represented as (x_min, x_max, y_min, y_max).
    - key_word (str): The word whose bounding box needs to be merged with the closest word to its right.
    - verbose (bool): If True, the function will print additional debug information.

    Returns:
    - list of tuple: An updated list of bounding boxes after merging the key_word with its following word.

    Note:
    If the key_word does not have a following word in the given list, the function will not make any changes.
    """
    try:
        data = find_next_right_word(converted_boxes, key_word, distance_margin=distance_margin,
                                    max_distance=max_distance, verbose=verbose)
        if verbose:
            print('data\n', data)
       #print(data)
        new_word = data['detected'][1] + ' ' + data['next'][1]
        new_box = (min(data['detected'][0][0], data['next'][0][0]),
                   max(data['detected'][0][1], data['next'][0][1]),
                   min(data['detected'][0][2], data['next'][0][2]),
                   max(data['detected'][0][3], data['next'][0][3]))

        converted_boxes.remove(data['detected'])
        converted_boxes.remove(data['next'])
        converted_boxes.append((new_box, new_word))

    except Exception as e:
        if not safe:
            raise e
        else:
            pass

    return converted_boxes

def remove_word(converted_boxes,
                word):
    return [(b,w) for (b,w) in converted_boxes if w != word]
